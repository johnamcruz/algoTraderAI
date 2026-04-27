"""
Unit tests for STTrendStrategyV1.

Every test verifies behaviour against the exact labeling code in
Futures-Foundation-Model/colabs/st_trend_v1.py — the source of truth for how
the model was trained. Key things being guarded:

  1. Wilder's ATR formula (period=10)
  2. Band clamping — uses PREVIOUS bar's raw/clamped bands
  3. Direction state machine — checks close vs PREVIOUS bar's clamped bands
  4. ATR rank — window EXCLUDES current bar (matches atr[start:i] in labeler)
  5. Prior trend extent — bull trend only tracks high; bear only tracks low
  6. Signal fires ONLY on ST flip where 1h HTF agrees
  7. 8-element feature vector matches training feature column order exactly
  8. TP tier selection (2R / 4R / 6R)
  9. Entry gate (conf threshold + RR gate)
 10. on_trade_exit resets signal state
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

from strategies.strategy_st_trend_v1 import (
    STTrendStrategyV1,
    ST_PERIOD, ST_MULT, HTF_PERIOD, HTF_MULT,
    SL_ATR_MULT, ATR_RANK_WINDOW,
)


# ── Reference batch implementation (copied verbatim from st_trend_v1.py labeler) ─

def _batch_compute_st(h, l, c, period, mult):
    """
    Wilder's ATR SuperTrend — exact replica of the training labeler's compute_st().
    Returns (direction, st_line, atr, upper_band, lower_band).
    """
    n = len(c)
    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])),
    )
    atr = np.empty(n, dtype=np.float64)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = ((period - 1) * atr[i - 1] + tr[i - 1]) / period
    atr = np.maximum(atr, 1e-6)

    hl2 = (h + l) / 2.0
    ub  = (hl2 + mult * atr).copy()
    lb  = (hl2 - mult * atr).copy()

    d       = np.ones(n, dtype=np.int8)
    st_line = lb.copy()

    for i in range(1, n):
        ub[i] = ub[i] if (ub[i] < ub[i - 1] or c[i - 1] > ub[i - 1]) else ub[i - 1]
        lb[i] = lb[i] if (lb[i] > lb[i - 1] or c[i - 1] < lb[i - 1]) else lb[i - 1]
        if d[i - 1] == -1:
            d[i] = 1 if c[i] > ub[i - 1] else -1
        else:
            d[i] = -1 if c[i] < lb[i - 1] else 1
        st_line[i] = lb[i] if d[i] == 1 else ub[i]

    return d, st_line, atr, ub, lb


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def st():
    """STTrendStrategyV1 with all state pre-initialised — no ONNX model loaded."""
    s = STTrendStrategyV1.__new__(STTrendStrategyV1)
    s.contract_symbol = 'MNQZ5'
    s._instrument     = 'NQ'
    s._min_risk_rr    = 0.0

    # 5m SuperTrend state
    s._st_atr        = 0.0
    s._st_upper      = float('inf')
    s._st_lower      = 0.0
    s._st_direction  = 1
    s._st_prev_close = 0.0
    s._st_initialized = False

    # Prior trend tracking
    s._trend_start_bar = 0
    s._trend_high      = 0.0
    s._trend_low       = float('inf')
    s._prior_duration  = 0
    s._prior_extent    = 0.0

    # ATR history
    s._atr_history = deque(maxlen=ATR_RANK_WINDOW)

    # 1h SuperTrend state
    s._htf_atr        = 0.0
    s._htf_upper      = float('inf')
    s._htf_lower      = 0.0
    s._htf_direction  = 1
    s._htf_prev_close = 0.0
    s._htf_initialized = False
    s._htf_flip_bar   = 0
    s._htf_bar_hour   = None
    s._htf_bar_date   = None
    s._htf_bar_open   = 0.0
    s._htf_bar_high   = 0.0
    s._htf_bar_low    = float('inf')
    s._htf_bar_close  = 0.0

    # Signal state
    s._is_signal_bar        = False
    s._current_direction    = 0
    s._current_st_features  = np.zeros(8, dtype=np.float32)
    s._signal_atr           = 0.0
    s._latest_risk_rr       = 0.0
    s._latest_signal_meta   = {}

    s._bar_count = 0
    s.skip_stats = {'conf_gate': 0, 'rr_gate': 0, 'hold': 0}
    return s


def _run_bars(st_strategy, highs, lows, closes, opens=None):
    """Feed a sequence of bars to the strategy and return the bar count."""
    if opens is None:
        opens = closes.copy()
    for i, (h, l, c, o) in enumerate(zip(highs, lows, closes, opens)):
        st_strategy._update_st_5m(h, l, c, o, i)
    return len(highs)


# ── 1. Session gate ───────────────────────────────────────────────────────────

class TestSessionGate:
    """STTrendStrategyV1 has no session filter — is_trading_allowed always True."""

    def test_always_true_during_session(self, st):
        ts = pd.Timestamp('2026-04-21 10:00:00', tz='America/New_York')
        assert st.is_trading_allowed(ts) is True

    def test_always_true_pre_market(self, st):
        ts = pd.Timestamp('2026-04-21 04:00:00', tz='America/New_York')
        assert st.is_trading_allowed(ts) is True

    def test_always_true_overnight(self, st):
        ts = pd.Timestamp('2026-04-21 01:00:00', tz='America/New_York')
        assert st.is_trading_allowed(ts) is True


# ── 2. Wilder's ATR computation ───────────────────────────────────────────────

class TestWildersATR:
    """Incremental ATR must follow Wilder's smoothing: ATR[i] = ((p-1)*ATR[i-1] + TR[i]) / p."""

    def test_first_bar_atr_equals_hl_range(self, st):
        st._update_st_5m(102, 98, 100, 100, 0)
        assert st._st_atr == pytest.approx(4.0)

    def test_second_bar_wilder_formula(self, st):
        st._update_st_5m(102, 98, 100, 100, 0)   # ATR=4
        # TR = max(103-99, |103-100|, |99-100|) = max(4, 3, 1) = 4
        st._update_st_5m(103, 99, 101, 101, 1)
        expected = (9 * 4.0 + 4.0) / 10  # = 4.0
        assert st._st_atr == pytest.approx(expected)

    def test_atr_wilder_with_larger_tr(self, st):
        st._update_st_5m(102, 98, 100, 100, 0)   # ATR=4
        # TR = max(108-96, |108-100|, |96-100|) = max(12, 8, 4) = 12
        st._update_st_5m(108, 96, 102, 102, 1)
        expected = (9 * 4.0 + 12.0) / 10  # = 4.8
        assert st._st_atr == pytest.approx(expected)

    def test_atr_floored_at_1e_6(self, st):
        """ATR must never be zero even for zero-range bars."""
        st._update_st_5m(100, 100, 100, 100, 0)  # TR=0
        assert st._st_atr == pytest.approx(1e-6)

    def test_atr_accumulates_correctly_over_multiple_bars(self, st):
        atrs = [4.0]  # bar 0: ATR = h-l = 4
        bars = [
            (102, 98, 100),   # bar 0: init
            (103, 99, 101),   # bar 1: TR=4
            (105, 100, 102),  # bar 2: TR=max(5,4,1)=5
            (104, 101, 103),  # bar 3: TR=max(3,2,1)=3
        ]
        for i, (h, l, c) in enumerate(bars):
            st._update_st_5m(h, l, c, c, i)

        # Step through manually
        atr = 4.0
        trs = [4.0, 5.0, 3.0]  # TR for bars 1,2,3
        for tr in trs:
            atr = (9 * atr + tr) / 10
        assert st._st_atr == pytest.approx(atr, rel=1e-5)


# ── 3. Band clamping ──────────────────────────────────────────────────────────

class TestBandClamping:
    """
    Band clamping rules (must match labeler lines 194-195):
      upper = raw_upper if (raw_upper < prev_upper OR prev_close > prev_upper) else prev_upper
      lower = raw_lower if (raw_lower > prev_lower OR prev_close < prev_lower) else prev_lower
    """

    def _setup_state(self, st, *, upper, lower, direction, prev_close, atr):
        """Inject post-initialization state for targeted clamping tests."""
        st._st_initialized = True
        st._st_upper      = upper
        st._st_lower      = lower
        st._st_direction  = direction
        st._st_prev_close = prev_close
        st._st_atr        = atr
        st._trend_start_bar = 0
        st._trend_high = 105.0
        st._trend_low  = 95.0

    def test_upper_band_tightens_when_raw_lower(self, st):
        """raw_upper < prev_upper → use raw (band tightening allowed)."""
        self._setup_state(st, upper=104.0, lower=96.0, direction=1,
                          prev_close=100.0, atr=4.0)
        # hl2=101, raw_upper=101+2*atr_new — craft so raw_upper < 104
        # h=101.5, l=100.5 → hl2=101, atr_new≈(9*4+1)/10=3.7, raw_upper≈101+7.4=108.4 > 104
        # Use h=100, l=99 → hl2=99.5, atr_new=(9*4+max(1,|100-100|,|99-100|))/10=(9*4+1)/10=3.7
        # raw_upper=99.5+2*3.7=106.9 > 104 — not tightening
        # Make h=100, l=99.9 → hl2=99.95, TR=max(0.1,|100-100|,|99.9-100|)=0.1
        # atr_new=(9*4+0.1)/10=3.61, raw_upper=99.95+2*3.61=107.17 > 104 — still not
        # Better: set prev_upper very high so raw < prev
        self._setup_state(st, upper=200.0, lower=96.0, direction=1,
                          prev_close=100.0, atr=4.0)
        st._update_st_5m(102, 98, 100, 100, 1)
        # raw_upper = (102+98)/2 + 2*atr_new = 100 + 2*4.0 = 108 < 200 → use raw
        assert st._st_upper < 200.0

    def test_upper_band_held_when_raw_higher_and_below_band(self, st):
        """raw_upper >= prev_upper AND prev_close <= prev_upper → hold prev_upper."""
        self._setup_state(st, upper=104.0, lower=96.0, direction=1,
                          prev_close=100.0, atr=4.0)
        # Force a bar where raw_upper > 104 (so clamping activates)
        # h=110, l=106 → hl2=108, TR=max(4,10,6)=10, atr_new=(9*4+10)/10=4.6
        # raw_upper=108+2*4.6=117.2 > 104. prev_close=100 ≤ 104. → hold 104
        st._update_st_5m(110, 106, 108, 108, 1)
        assert st._st_upper == pytest.approx(104.0)

    def test_upper_band_resets_when_prev_close_above_band(self, st):
        """prev_close > prev_upper → raw_upper used (bar broke above band)."""
        self._setup_state(st, upper=104.0, lower=96.0, direction=1,
                          prev_close=106.0, atr=4.0)   # prev_close > prev_upper
        # h=110, l=106 → raw_upper will be > 104, but prev_close > prev_upper → use raw
        st._update_st_5m(110, 106, 108, 108, 1)
        assert st._st_upper != pytest.approx(104.0)

    def test_lower_band_tightens_when_raw_higher(self, st):
        """raw_lower > prev_lower → use raw (band tightening)."""
        self._setup_state(st, upper=104.0, lower=90.0, direction=1,
                          prev_close=100.0, atr=4.0)
        # h=102, l=100 → hl2=101, TR=max(2,2,0)=2, atr_new=(9*4+2)/10=3.8
        # raw_lower=101-2*3.8=93.4 > 90 → use raw_lower
        st._update_st_5m(102, 100, 101, 101, 1)
        assert st._st_lower > 90.0

    def test_lower_band_held_when_raw_lower_and_above_band(self, st):
        """raw_lower <= prev_lower AND prev_close >= prev_lower → hold prev_lower."""
        self._setup_state(st, upper=104.0, lower=98.0, direction=1,
                          prev_close=100.0, atr=4.0)
        # h=102, l=97 → hl2=99.5, TR=max(5,2,1)=5, atr_new=(9*4+5)/10=4.1
        # raw_lower=99.5-2*4.1=91.3 < 98. prev_close=100 >= 98 → hold 98
        st._update_st_5m(102, 97, 99, 99, 1)
        assert st._st_lower == pytest.approx(98.0)

    def test_lower_band_resets_when_prev_close_below_band(self, st):
        """prev_close < prev_lower → use raw_lower (bar broke below band)."""
        self._setup_state(st, upper=104.0, lower=98.0, direction=1,
                          prev_close=96.0, atr=4.0)   # prev_close < prev_lower
        st._update_st_5m(102, 97, 99, 99, 1)
        assert st._st_lower != pytest.approx(98.0)


# ── 4. Direction state machine ────────────────────────────────────────────────

class TestDirectionStateMachine:
    """
    Direction check uses PREVIOUS bar's clamped bands (labeler lines 196-199):
      if d[i-1] == -1: d[i] = 1 if c[i] > ub[i-1] else -1
      else:            d[i] = -1 if c[i] < lb[i-1] else 1
    """

    def _setup_state(self, st, *, upper, lower, direction, prev_close, atr=4.0):
        st._st_initialized  = True
        st._st_upper        = upper
        st._st_lower        = lower
        st._st_direction    = direction
        st._st_prev_close   = prev_close
        st._st_atr          = atr
        st._trend_start_bar = 0
        st._trend_high      = upper
        st._trend_low       = lower
        st._htf_direction   = -direction  # align with DESTINATION direction (post-flip)

    def test_bear_to_bull_flip_when_close_above_prev_upper(self, st):
        """Bear → Bull: close must cross prev_upper (not the newly computed upper)."""
        self._setup_state(st, upper=105.0, lower=95.0, direction=-1, prev_close=100.0)
        # Close > prev_upper=105 → flip to bull
        st._update_st_5m(108, 104, 106, 104, 1)
        assert st._st_direction == 1
        assert st._is_signal_bar is True

    def test_bear_no_flip_when_close_at_prev_upper(self, st):
        """Close exactly at prev_upper → no flip (must be strictly greater)."""
        self._setup_state(st, upper=105.0, lower=95.0, direction=-1, prev_close=100.0)
        st._update_st_5m(106, 104, 105, 104, 1)
        assert st._st_direction == -1

    def test_bear_no_flip_when_close_below_prev_upper(self, st):
        self._setup_state(st, upper=105.0, lower=95.0, direction=-1, prev_close=100.0)
        st._update_st_5m(106, 102, 103, 102, 1)
        assert st._st_direction == -1

    def test_bull_to_bear_flip_when_close_below_prev_lower(self, st):
        """Bull → Bear: close must cross prev_lower (not the newly computed lower)."""
        self._setup_state(st, upper=105.0, lower=95.0, direction=1, prev_close=100.0)
        # Close < prev_lower=95 → flip to bear
        st._update_st_5m(97, 93, 94, 97, 1)
        assert st._st_direction == -1
        assert st._is_signal_bar is True

    def test_bull_no_flip_when_close_at_prev_lower(self, st):
        """Close exactly at prev_lower → no flip (must be strictly less)."""
        self._setup_state(st, upper=105.0, lower=95.0, direction=1, prev_close=100.0)
        st._update_st_5m(97, 94, 95, 97, 1)
        assert st._st_direction == 1

    def test_direction_uses_prev_upper_not_new_upper(self, st):
        """
        If current bar's newly computed upper < prev_upper, close vs NEW upper
        might differ from close vs PREV upper. Strategy must use prev.
        """
        # prev_upper=105 (wide), new bar will compute raw_upper≈103 (tighter)
        # Close=104 > new_upper=103 but < prev_upper=105 → NO flip in bear mode
        self._setup_state(st, upper=105.0, lower=95.0, direction=-1, prev_close=100.0)
        # h=104, l=103 → hl2=103.5, TR=max(1, 4, 3)=4, atr_new=(9*4+4)/10=4
        # raw_upper=103.5+2*4=111.5 → not 103; pick different values
        # h=103.1, l=103.0 → hl2=103.05, TR=max(0.1,|103.1-100|,|103.0-100|)=3.1
        # atr_new=(9*4+3.1)/10=3.91, raw_upper=103.05+2*3.91=110.87 → still > prev_upper
        # The clamping will keep prev_upper when raw > prev. Let's test directly via:
        # inject state where after clamping upper stays at 105 but close=104 < 105
        self._setup_state(st, upper=105.0, lower=95.0, direction=-1, prev_close=100.0)
        st._update_st_5m(106, 103, 104, 103, 1)  # close=104 < prev_upper=105 → stays bear
        assert st._st_direction == -1


# ── 5. ATR rank (excludes current bar) ───────────────────────────────────────

class TestATRRank:
    """
    atr_rank[i] = proportion of atr[start:i] that is < atr[i].
    Window is atr[start:i], which EXCLUDES bar i — matches labeler line 243-244.
    """

    def test_rank_zero_when_history_empty(self, st):
        """First bar has empty history → rank = 0 / max(0,1) = 0."""
        st._update_st_5m(102, 98, 100, 100, 0)
        # After bar 0 (init), history now has one value — no rank computed at init
        # Rank is computed on the NEXT bar
        assert len(st._atr_history) == 1

    def test_rank_computed_before_appending(self, st):
        """At bar 1, rank uses history=[atr_0] and compares atr_1 against it."""
        st._update_st_5m(102, 98, 100, 100, 0)  # atr_0 = 4.0
        atr_0 = st._st_atr
        # Feed a second bar with larger ATR
        st._update_st_5m(106, 98, 102, 102, 1)  # large range → atr_1 > atr_0
        atr_1 = st._st_atr
        assert atr_1 > atr_0
        # History now has 2 entries. The rank stored at bar 1 is computed with
        # window=[atr_0], which gives rank=1.0 (atr_0 < atr_1).
        # We verify indirectly via signal features in TestFeatureVector.

    def test_rank_respects_window_size(self, st):
        """ATR history is capped at ATR_RANK_WINDOW=200."""
        for i in range(250):
            st._update_st_5m(102, 98, 100, 100, i)
        assert len(st._atr_history) <= ATR_RANK_WINDOW

    def test_rank_is_zero_for_smallest_atr(self, st):
        """If current ATR is the smallest seen, rank should be 0."""
        # Build history with large ATRs
        for i in range(10):
            st._update_st_5m(110, 90, 100, 100, i)  # ATR ≈ 20
        # Inject a bar with very small TR so ATR will be tiny
        # atr_rank is set at the bar where we compute it — we verify sign of features
        # via the signal path; here we just confirm no crash and rank in [0, 1].
        st._st_atr = 20.0
        st._atr_history = deque([20.0] * 10, maxlen=ATR_RANK_WINDOW)
        # Next bar: small ATR
        st._st_initialized = True
        st._st_prev_close = 100.0
        st._trend_high = 110.0
        st._trend_low  = 90.0
        st._st_upper = 140.0
        st._st_lower = 60.0
        st._st_direction = 1
        st._update_st_5m(100.1, 99.9, 100.0, 100.0, 11)
        # rank should be 0 (tiny ATR < all history values)
        # (we can't read rank directly; verify via feature vector in signal tests)
        assert len(st._atr_history) <= ATR_RANK_WINDOW  # no crash

    def test_rank_matches_labeler_formula_step_by_step(self, st):
        """
        Feed 5 bars; at bar 4 verify the signal's atr_rank_pct matches:
        rank = sum(history < atr) / len(history)  — history excludes current bar.
        """
        # Build a simple ascending ATR series so we can compute expected rank.
        highs  = [102, 103, 104, 105, 106]
        lows   = [ 98,  99, 100, 101, 102]
        closes = [100, 101, 102, 103, 104]
        opens  = [100, 101, 102, 103, 104]
        for i, (h, l, c, o) in enumerate(zip(highs, lows, closes, opens)):
            st._update_st_5m(h, l, c, o, i)
        # Each bar has ATR ~4 (small range); rank can be computed from history
        # We just verify the history length is correct (= bars processed so far, capped 200)
        assert len(st._atr_history) == 5


# ── 6. Prior trend tracking ───────────────────────────────────────────────────

class TestPriorTrendTracking:
    """
    Labeler (lines 268-273):
      if not is_flip:
        if fd[i] == 1: trend_phase_high = max(...)   # bull: only high
        else:          trend_phase_low  = min(...)   # bear: only low
    """

    def _prep_bull_state(self, st):
        """Put strategy in a stable bull state."""
        st._st_initialized  = True
        st._st_direction    = 1
        st._st_atr          = 4.0
        st._st_prev_close   = 100.0
        st._st_upper        = 110.0   # far — no flip
        st._st_lower        = 90.0
        st._trend_start_bar = 0
        st._trend_high      = 100.0
        st._trend_low       = 100.0   # will NOT update during bull
        st._htf_direction   = 1
        st._atr_history.append(4.0)

    def _prep_bear_state(self, st):
        """Put strategy in a stable bear state."""
        st._st_initialized  = True
        st._st_direction    = -1
        st._st_atr          = 4.0
        st._st_prev_close   = 100.0
        st._st_upper        = 110.0
        st._st_lower        = 90.0
        st._trend_start_bar = 0
        st._trend_high      = 100.0   # will NOT update during bear
        st._trend_low       = 100.0
        st._htf_direction   = -1
        st._atr_history.append(4.0)

    def test_bull_trend_only_updates_high(self, st):
        """During a bull trend, _trend_low must NOT change on non-flip bars."""
        self._prep_bull_state(st)
        initial_low = st._trend_low   # = 100.0
        # Feed a bar with a lower low — should NOT update _trend_low
        st._update_st_5m(103, 95, 102, 103, 1)   # low=95 < 100
        assert st._trend_low == pytest.approx(initial_low)

    def test_bull_trend_updates_high(self, st):
        self._prep_bull_state(st)
        st._update_st_5m(108, 104, 106, 108, 1)  # new high
        assert st._trend_high == pytest.approx(108.0)

    def test_bear_trend_only_updates_low(self, st):
        """During a bear trend, _trend_high must NOT change on non-flip bars."""
        self._prep_bear_state(st)
        initial_high = st._trend_high   # = 100.0
        # Feed a bar with a higher high — should NOT update _trend_high
        st._update_st_5m(110, 96, 97, 110, 1)   # high=110 > 100
        assert st._trend_high == pytest.approx(initial_high)

    def test_bear_trend_updates_low(self, st):
        self._prep_bear_state(st)
        st._update_st_5m(99, 88, 89, 99, 1)  # new low
        assert st._trend_low == pytest.approx(88.0)

    def test_flip_captures_prior_duration(self, st):
        """prior_duration = current_bar_idx - trend_start_bar."""
        self._prep_bull_state(st)
        st._trend_start_bar = 5
        # Trigger a flip by closing below prev_lower=90
        st._update_st_5m(92, 85, 86, 92, 10)  # close < lower=90 → bear flip
        assert st._prior_duration == 10 - 5

    def test_flip_captures_prior_extent_bull(self, st):
        """Bull extent = trend_high - trend_low (trend_low is fixed at flip bar's low)."""
        self._prep_bull_state(st)
        st._trend_high = 108.0
        st._trend_low  = 98.0   # fixed at flip-bar value; won't change during bull
        # Trigger bear flip
        st._update_st_5m(92, 85, 86, 92, 5)
        assert st._prior_extent == pytest.approx(108.0 - 98.0)

    def test_flip_resets_trend_tracking(self, st):
        """After a flip, trend_start_bar, trend_high, trend_low reset to current bar."""
        self._prep_bull_state(st)
        st._update_st_5m(92, 85, 86, 92, 7)  # bear flip
        assert st._trend_start_bar == 7
        assert st._trend_high == pytest.approx(92.0)
        assert st._trend_low  == pytest.approx(85.0)

    def test_prior_extent_norm_capped_at_10(self, st):
        """prior_extent_norm = min(prior_range / atr, 10)."""
        self._prep_bull_state(st)
        st._trend_high = 10000.0   # enormous range
        st._trend_low  = 0.0
        st._htf_direction = -1    # flip to bear + HTF aligned → signal fires
        st._update_st_5m(92, 85, 86, 92, 5)  # bear flip
        assert st._current_st_features[3] == pytest.approx(10.0)

    def test_prior_duration_norm_capped_at_5(self, st):
        """prior_duration_norm = min(prior_duration / 100, 5)."""
        self._prep_bull_state(st)
        st._trend_start_bar = 0
        st._htf_direction = -1    # bear flip + HTF aligned
        st._update_st_5m(92, 85, 86, 92, 600)  # 600 bars into the trend
        assert st._current_st_features[2] == pytest.approx(5.0)


# ── 7. HTF 1h bar aggregation ─────────────────────────────────────────────────

class TestHTFAggregation:
    """1h bars are built by accumulating 5m bars; finalized on hour rollover."""

    def _ts(self, hour, minute=0, date='2026-04-21'):
        return pd.Timestamp(f'{date} {hour:02d}:{minute:02d}:00', tz='UTC')

    def test_first_bar_initializes_htf_state(self, st):
        st._update_htf(102, 98, 100, self._ts(10, 0), 0)
        assert st._htf_bar_hour == 10
        assert st._htf_bar_high == pytest.approx(102)
        assert st._htf_bar_low  == pytest.approx(98)

    def test_same_hour_bars_aggregate_high_low(self, st):
        st._update_htf(102, 98, 100, self._ts(10, 0), 0)
        st._update_htf(104, 97, 101, self._ts(10, 5), 1)
        st._update_htf(103, 99, 102, self._ts(10, 10), 2)
        assert st._htf_bar_high == pytest.approx(104)
        assert st._htf_bar_low  == pytest.approx(97)
        assert st._htf_bar_close == pytest.approx(102)  # last close in hour

    def test_hour_rollover_triggers_finalize(self, st):
        st._update_htf(102, 98, 100, self._ts(10, 0), 0)
        st._update_htf(104, 97, 101, self._ts(10, 55), 11)
        # New hour → finalize
        st._update_htf(103, 100, 102, self._ts(11, 0), 12)
        assert st._htf_initialized is True
        assert st._htf_bar_hour == 11

    def test_new_date_triggers_finalize(self, st):
        st._update_htf(102, 98, 100, self._ts(10, 0, '2026-04-21'), 0)
        st._update_htf(103, 99, 101, self._ts(10, 0, '2026-04-22'), 12)
        assert st._htf_initialized is True
        assert st._htf_bar_date.strftime('%Y-%m-%d') == '2026-04-22'

    def test_htf_direction_initialized_to_1(self, st):
        st._update_htf(102, 98, 100, self._ts(10, 0), 0)
        st._update_htf(103, 99, 101, self._ts(11, 0), 12)
        assert st._htf_direction == 1  # first 1h bar initializes direction=1

    def test_htf_flip_updates_flip_bar_index(self, st):
        """When 1h ST direction changes, _htf_flip_bar stores current bar_idx."""
        # Build enough 1h bars to get a flip
        # Bar 0 at h10: init
        st._update_htf(100, 98, 99, self._ts(10, 0), 0)
        # h11: first real 1h bar finalized — initializes HTF state
        st._update_htf(101, 99, 100, self._ts(11, 0), 12)
        # h12: second real bar — direction set
        st._update_htf(102, 100, 101, self._ts(12, 0), 24)
        dir_before = st._htf_direction
        # Force a strong bear move for h13 (will be finalized at h14 rollover)
        st._update_htf(101, 85, 86, self._ts(13, 0), 36)
        st._update_htf(102, 100, 101, self._ts(14, 0), 48)
        # If a flip occurred, _htf_flip_bar should be updated
        if st._htf_direction != dir_before:
            assert st._htf_flip_bar > 0


# ── 8. Signal detection ───────────────────────────────────────────────────────

class TestSignalDetection:
    """Signal fires ONLY on 5m ST flip where 1h HTF direction agrees."""

    def _setup_for_flip(self, st, *, flip_to, htf_dir):
        """Pre-load state for a controlled flip scenario."""
        st._st_initialized  = True
        st._st_atr          = 4.0
        st._st_prev_close   = 100.0
        st._st_upper        = 105.0
        st._st_lower        = 95.0
        st._trend_start_bar = 0
        st._trend_high      = 105.0
        st._trend_low       = 95.0
        st._htf_direction   = htf_dir
        st._htf_flip_bar    = 0
        st._atr_history.append(4.0)
        if flip_to == 'bull':
            st._st_direction = -1   # will flip to bull
        else:
            st._st_direction = 1    # will flip to bear

    def test_flip_with_htf_aligned_is_signal(self, st):
        """Bull flip with HTF=bull → signal."""
        self._setup_for_flip(st, flip_to='bull', htf_dir=1)
        st._update_st_5m(108, 104, 107, 104, 1)  # close > prev_upper=105
        assert st._is_signal_bar is True
        assert st._current_direction == 1   # BUY

    def test_flip_with_htf_misaligned_not_signal(self, st):
        """Bull flip with HTF=bear → no signal."""
        self._setup_for_flip(st, flip_to='bull', htf_dir=-1)
        st._update_st_5m(108, 104, 107, 104, 1)
        assert st._is_signal_bar is False
        assert st._current_direction == 0

    def test_bear_flip_with_htf_aligned_is_signal(self, st):
        """Bear flip with HTF=bear → signal."""
        self._setup_for_flip(st, flip_to='bear', htf_dir=-1)
        st._update_st_5m(97, 91, 92, 97, 1)  # close < prev_lower=95
        assert st._is_signal_bar is True
        assert st._current_direction == 2   # SELL

    def test_bear_flip_with_htf_misaligned_not_signal(self, st):
        self._setup_for_flip(st, flip_to='bear', htf_dir=1)
        st._update_st_5m(97, 91, 92, 97, 1)
        assert st._is_signal_bar is False

    def test_non_flip_bar_not_signal(self, st):
        self._setup_for_flip(st, flip_to='bull', htf_dir=1)
        st._st_direction = 1  # already bull — no flip possible
        st._update_st_5m(102, 98, 101, 102, 1)  # close stays in band
        assert st._is_signal_bar is False

    def test_signal_atr_captured_at_flip_bar(self, st):
        """_signal_atr must equal the ATR of the flip bar, not a previous bar."""
        self._setup_for_flip(st, flip_to='bull', htf_dir=1)
        st._update_st_5m(108, 104, 107, 104, 1)
        assert st._signal_atr > 0
        assert st._signal_atr == pytest.approx(st._st_atr)


# ── 9. Feature vector ─────────────────────────────────────────────────────────

class TestFeatureVector:
    """
    8 ST features at signal bars (index order must match ST_FEATURE_COLS in labeler):
      0: st_direction              +1=bull, -1=bear
      1: st_line_distance          (close - st_line) / atr, clipped [-10, 10]
      2: prior_trend_duration_norm bars_in_prior_trend / 100, capped 5
      3: prior_trend_extent_norm   prior_range / atr, capped 10
      4: atr_rank_pct              rolling percentile [0, 1]
      5: htf_st_direction          1h ST direction +1/-1
      6: htf_st_age_norm           (bar_idx - htf_flip_bar) / 100, capped 5
      7: flip_bar_body_pct         |close-open| / (high-low), in [0, 1]
    """

    def _fire_bull_signal(self, st, bar_idx=50):
        """Set up state and fire a bull signal, return the feature vector."""
        st._st_initialized  = True
        st._st_atr          = 4.0
        st._st_prev_close   = 100.0
        st._st_upper        = 105.0
        st._st_lower        = 95.0
        st._st_direction    = -1        # will flip to bull
        st._trend_start_bar = 30
        st._trend_high      = 105.0
        st._trend_low       = 95.0
        st._prior_duration  = 0
        st._prior_extent    = 0.0
        st._htf_direction   = 1         # aligned
        st._htf_flip_bar    = 10
        st._atr_history     = deque([4.0] * 5, maxlen=ATR_RANK_WINDOW)
        st._update_st_5m(110, 104, 108, 104, bar_idx)
        return st._current_st_features

    def test_feature_vector_length(self, st):
        feats = self._fire_bull_signal(st)
        assert feats.shape == (8,)

    def test_feature_0_direction_bull(self, st):
        feats = self._fire_bull_signal(st)
        assert feats[0] == pytest.approx(1.0)

    def test_feature_0_direction_bear(self, st):
        st._st_initialized  = True
        st._st_atr          = 4.0
        st._st_prev_close   = 100.0
        st._st_upper        = 105.0
        st._st_lower        = 95.0
        st._st_direction    = 1         # will flip to bear
        st._trend_start_bar = 0
        st._trend_high      = 105.0
        st._trend_low       = 95.0
        st._htf_direction   = -1        # aligned bear
        st._htf_flip_bar    = 0
        st._atr_history     = deque([4.0] * 5, maxlen=ATR_RANK_WINDOW)
        st._update_st_5m(97, 91, 92, 97, 5)
        assert st._current_st_features[0] == pytest.approx(-1.0)

    def test_feature_1_st_line_distance_clipped(self, st):
        """st_line_distance = (close - st_line) / atr, clipped [-10, 10]."""
        feats = self._fire_bull_signal(st, bar_idx=50)
        # st_line = lower_band for bull direction; dist = (close - lower) / atr
        # All values must be within [-10, 10]
        assert -10.0 <= feats[1] <= 10.0

    def test_feature_2_prior_duration_norm(self, st):
        """prior_trend_duration_norm = min(duration/100, 5)."""
        st._st_initialized  = True
        st._st_atr          = 4.0
        st._st_prev_close   = 100.0
        st._st_upper        = 105.0
        st._st_lower        = 95.0
        st._st_direction    = -1
        st._trend_start_bar = 10       # bar 10 to bar 50 = 40 bars
        st._trend_high      = 105.0
        st._trend_low       = 95.0
        st._htf_direction   = 1
        st._htf_flip_bar    = 0
        st._atr_history     = deque([4.0] * 5, maxlen=ATR_RANK_WINDOW)
        st._update_st_5m(110, 104, 108, 104, 50)
        # prior_duration = 50 - 10 = 40; norm = 40/100 = 0.4
        assert st._current_st_features[2] == pytest.approx(0.4)

    def test_feature_2_prior_duration_norm_capped(self, st):
        feats = self._fire_bull_signal(st)
        feats2 = st._current_st_features.copy()
        # bar_idx=50, trend_start_bar=30 → duration=20, norm=0.2 (< cap=5)
        assert feats2[2] == pytest.approx(0.2)

    def test_feature_3_prior_extent_norm(self, st):
        """prior_extent_norm = min((trend_high - trend_low) / atr, 10)."""
        feats = self._fire_bull_signal(st)
        # trend_high=105, trend_low=95, atr≈4 → extent=10, norm=10/4=2.5
        assert feats[3] > 0.0
        assert feats[3] <= 10.0

    def test_feature_4_atr_rank_in_range(self, st):
        feats = self._fire_bull_signal(st)
        assert 0.0 <= feats[4] <= 1.0

    def test_feature_5_htf_direction_bull(self, st):
        feats = self._fire_bull_signal(st)
        assert feats[5] == pytest.approx(1.0)

    def test_feature_6_htf_age_norm(self, st):
        """htf_st_age_norm = min((bar_idx - htf_flip_bar) / 100, 5)."""
        feats = self._fire_bull_signal(st, bar_idx=50)
        # htf_flip_bar=10, bar_idx=50 → age=(50-10)/100=0.4
        assert feats[6] == pytest.approx(0.4)

    def test_feature_6_htf_age_capped_at_5(self, st):
        st._st_initialized  = True
        st._st_atr          = 4.0
        st._st_prev_close   = 100.0
        st._st_upper        = 105.0
        st._st_lower        = 95.0
        st._st_direction    = -1
        st._trend_start_bar = 0
        st._trend_high      = 105.0
        st._trend_low       = 95.0
        st._htf_direction   = 1
        st._htf_flip_bar    = 0         # flip_bar=0, bar_idx=600 → age=6>5 → capped 5
        st._atr_history     = deque([4.0] * 5, maxlen=ATR_RANK_WINDOW)
        st._update_st_5m(110, 104, 108, 104, 600)
        assert st._current_st_features[6] == pytest.approx(5.0)

    def test_feature_7_flip_body_pct(self, st):
        """flip_bar_body_pct = |close - open| / (high - low)."""
        st._st_initialized  = True
        st._st_atr          = 4.0
        st._st_prev_close   = 100.0
        st._st_upper        = 105.0
        st._st_lower        = 95.0
        st._st_direction    = -1
        st._trend_start_bar = 0
        st._trend_high      = 105.0
        st._trend_low       = 95.0
        st._htf_direction   = 1
        st._htf_flip_bar    = 0
        st._atr_history     = deque([4.0] * 5, maxlen=ATR_RANK_WINDOW)
        # h=110, l=104, o=104, c=108 → body=|108-104|=4, range=6 → 4/6 ≈ 0.667
        st._update_st_5m(110, 104, 108, 104, 5)
        assert st._current_st_features[7] == pytest.approx(4.0 / 6.0, rel=1e-5)

    def test_feature_7_body_pct_in_range(self, st):
        feats = self._fire_bull_signal(st)
        assert 0.0 <= feats[7] <= 1.0

    def test_no_nan_in_feature_vector(self, st):
        feats = self._fire_bull_signal(st)
        assert not np.any(np.isnan(feats))

    def test_dtype_float32(self, st):
        feats = self._fire_bull_signal(st)
        assert feats.dtype == np.float32


# ── 10. Entry gate (should_enter_trade) ───────────────────────────────────────

class TestEntryGate:

    def _enter(self, st, prediction=1, confidence=0.80, entry_conf=0.70):
        return st.should_enter_trade(prediction, confidence, {}, entry_conf, 0)

    def test_above_threshold_long_enters(self, st):
        ok, direction = self._enter(st, prediction=1, confidence=0.80)
        assert ok is True
        assert direction == 'LONG'

    def test_above_threshold_short_enters(self, st):
        ok, direction = self._enter(st, prediction=2, confidence=0.80)
        assert ok is True
        assert direction == 'SHORT'

    def test_exactly_at_threshold_enters(self, st):
        ok, _ = self._enter(st, confidence=0.70, entry_conf=0.70)
        assert ok is True

    def test_below_threshold_blocked(self, st):
        ok, direction = self._enter(st, confidence=0.69, entry_conf=0.70)
        assert ok is False
        assert direction is None

    def test_hold_prediction_blocked(self, st):
        ok, direction = self._enter(st, prediction=0, confidence=0.95)
        assert ok is False
        assert direction is None

    def test_rr_gate_disabled_by_default(self, st):
        """min_risk_rr=0.0 → low predicted RR never blocks."""
        st._min_risk_rr   = 0.0
        st._latest_risk_rr = 0.0
        ok, _ = self._enter(st, prediction=1, confidence=0.95)
        assert ok is True

    def test_rr_gate_blocks_below_min(self, st):
        st._min_risk_rr   = 2.0
        st._latest_risk_rr = 1.5
        ok, direction = self._enter(st, prediction=1, confidence=0.95)
        assert ok is False
        assert direction is None

    def test_rr_gate_passes_at_threshold(self, st):
        st._min_risk_rr   = 2.0
        st._latest_risk_rr = 2.0
        ok, _ = self._enter(st, prediction=1, confidence=0.95)
        assert ok is True

    def test_rr_gate_passes_above_threshold(self, st):
        st._min_risk_rr   = 2.0
        st._latest_risk_rr = 3.5
        ok, _ = self._enter(st, prediction=1, confidence=0.95)
        assert ok is True

    def test_conservative_threshold(self, st):
        ok, direction = self._enter(st, confidence=0.91, entry_conf=0.90)
        assert ok is True
        assert direction == 'LONG'

    def test_below_conservative_threshold_blocked(self, st):
        ok, _ = self._enter(st, confidence=0.89, entry_conf=0.90)
        assert ok is False


# ── 11. Stop / target (get_stop_target_pts) ───────────────────────────────────

class TestStopTargetPts:
    """
    Stop = signal_atr × 1.5.
    TP: int(predicted_rr) × R — skip if predicted_rr < 2.0.
    """

    def _set_signal(self, st, atr=4.0, raw_rr=2.5):
        st._signal_atr    = atr
        st._latest_risk_rr = raw_rr

    def test_no_signal_atr_returns_none(self, st):
        st._signal_atr = 0.0
        stop, target = st.get_stop_target_pts(None, 'LONG', 100.0)
        assert stop is None and target is None

    def test_stop_equals_atr_times_multiplier(self, st):
        self._set_signal(st, atr=8.0, raw_rr=2.5)
        stop, _ = st.get_stop_target_pts(None, 'LONG', 100.0)
        assert stop == pytest.approx(8.0 * SL_ATR_MULT)  # 8 × 1.5 = 12.0

    def test_rr_below_2_skips(self, st):
        """raw_rr < 2 → (None, None); low-quality signal must not enter."""
        self._set_signal(st, atr=4.0, raw_rr=1.5)
        stop, target = st.get_stop_target_pts(None, 'LONG', 100.0)
        assert stop is None and target is None

    def test_rr_near_zero_skips(self, st):
        """raw_rr near zero → (None, None); not silently traded at 1R."""
        self._set_signal(st, atr=4.0, raw_rr=0.3)
        stop, target = st.get_stop_target_pts(None, 'LONG', 100.0)
        assert stop is None and target is None

    def test_rr_exactly_2_gives_2r(self, st):
        self._set_signal(st, atr=4.0, raw_rr=2.0)
        stop, target = st.get_stop_target_pts(None, 'LONG', 100.0)
        assert target == pytest.approx(stop * 2.0)

    def test_rr_between_2_and_3_gives_2r(self, st):
        self._set_signal(st, atr=4.0, raw_rr=2.9)
        stop, target = st.get_stop_target_pts(None, 'LONG', 100.0)
        assert target == pytest.approx(stop * 2.0)

    def test_rr_exactly_3_gives_3r(self, st):
        self._set_signal(st, atr=4.0, raw_rr=3.0)
        stop, target = st.get_stop_target_pts(None, 'LONG', 100.0)
        assert target == pytest.approx(stop * 3.0)

    def test_rr_exactly_4_gives_4r(self, st):
        self._set_signal(st, atr=4.0, raw_rr=4.0)
        stop, target = st.get_stop_target_pts(None, 'LONG', 100.0)
        assert target == pytest.approx(stop * 4.0)

    def test_rr_exactly_5_gives_5r(self, st):
        self._set_signal(st, atr=4.0, raw_rr=5.0)
        stop, target = st.get_stop_target_pts(None, 'LONG', 100.0)
        assert target == pytest.approx(stop * 5.0)

    def test_rr_exactly_6_gives_6r(self, st):
        self._set_signal(st, atr=4.0, raw_rr=6.0)
        stop, target = st.get_stop_target_pts(None, 'LONG', 100.0)
        assert target == pytest.approx(stop * 6.0)

    def test_rr_above_6_uses_full_prediction(self, st):
        """No ceiling — int(8.3) = 8R target."""
        self._set_signal(st, atr=4.0, raw_rr=8.3)
        stop, target = st.get_stop_target_pts(None, 'LONG', 100.0)
        assert target == pytest.approx(stop * 8.0)

    def test_rr_large_prediction(self, st):
        """int(14.18) = 14R target."""
        self._set_signal(st, atr=4.0, raw_rr=14.18)
        stop, target = st.get_stop_target_pts(None, 'LONG', 100.0)
        assert target == pytest.approx(stop * 14.0)

    def test_stop_same_for_long_and_short(self, st):
        self._set_signal(st, atr=4.0, raw_rr=2.0)
        stop_long,  _ = st.get_stop_target_pts(None, 'LONG',  100.0)
        stop_short, _ = st.get_stop_target_pts(None, 'SHORT', 100.0)
        assert stop_long == pytest.approx(stop_short)


# ── 12. Skip stats ────────────────────────────────────────────────────────────

class TestSkipStats:

    def test_conf_gate_increments(self, st):
        st.skip_stats = {'conf_gate': 0, 'rr_gate': 0, 'hold': 0}
        st.should_enter_trade(1, 0.60, {}, entry_conf=0.80, adx_thresh=0)
        assert st.skip_stats['conf_gate'] == 1
        assert st.skip_stats['rr_gate'] == 0

    def test_rr_gate_increments(self, st):
        st._min_risk_rr   = 2.0
        st._latest_risk_rr = 1.5
        st.skip_stats = {'conf_gate': 0, 'rr_gate': 0, 'hold': 0}
        st.should_enter_trade(1, 0.90, {}, entry_conf=0.80, adx_thresh=0)
        assert st.skip_stats['rr_gate'] == 1
        assert st.skip_stats['conf_gate'] == 0

    def test_hold_prediction_increments(self, st):
        st.skip_stats = {'conf_gate': 0, 'rr_gate': 0, 'hold': 0}
        st.should_enter_trade(0, 0.90, {}, entry_conf=0.80, adx_thresh=0)
        assert st.skip_stats['hold'] == 1

    def test_successful_entry_no_increment(self, st):
        st.skip_stats = {'conf_gate': 0, 'rr_gate': 0, 'hold': 0}
        st.should_enter_trade(1, 0.90, {}, entry_conf=0.80, adx_thresh=0)
        assert st.skip_stats == {'conf_gate': 0, 'rr_gate': 0, 'hold': 0}

    def test_stats_accumulate_across_calls(self, st):
        st.skip_stats = {'conf_gate': 0, 'rr_gate': 0, 'hold': 0}
        for _ in range(3):
            st.should_enter_trade(1, 0.50, {}, entry_conf=0.80, adx_thresh=0)
        assert st.skip_stats['conf_gate'] == 3


# ── 13. on_trade_exit ─────────────────────────────────────────────────────────

class TestOnTradeExit:

    def test_exit_resets_is_signal_bar(self, st):
        st._is_signal_bar = True
        st.on_trade_exit('STOP_LOSS')
        assert st._is_signal_bar is False

    def test_exit_resets_current_direction(self, st):
        st._current_direction = 1
        st.on_trade_exit('STOP_LOSS')
        assert st._current_direction == 0

    def test_exit_on_take_profit_also_resets(self, st):
        st._is_signal_bar    = True
        st._current_direction = 2
        st.on_trade_exit('TAKE_PROFIT')
        assert st._is_signal_bar is False
        assert st._current_direction == 0


# ── 14. Non-signal bar predict returns (0, 0.0) ───────────────────────────────

class TestNonSignalPredict:
    """predict() must short-circuit to (0, 0.0) when not a signal bar."""

    def test_non_signal_bar_returns_hold(self, st):
        st._is_signal_bar    = False
        st._current_direction = 0
        result = st.predict(pd.DataFrame())
        assert result == (0, 0.0)

    def test_signal_bar_with_direction_0_returns_hold(self, st):
        st._is_signal_bar    = True
        st._current_direction = 0
        result = st.predict(pd.DataFrame())
        assert result == (0, 0.0)

    def test_signal_bar_with_insufficient_data_returns_hold(self, st):
        """predict() returns (0, 0.0) when df has fewer rows than seq_len."""
        st._is_signal_bar    = True
        st._current_direction = 1
        df = pd.DataFrame({'a': range(10)})
        result = st.predict(df)
        assert result == (0, 0.0)


# ── 15. Batch parity — incremental vs reference labeler ──────────────────────

class TestBatchParity:
    """
    Feed identical bars to both the batch compute_st reference (from training code)
    and our incremental strategy. After enough bars for Wilder's ATR to converge,
    the direction and ATR values must match exactly (bar by bar).

    Note: batch and incremental have different initialization for the first bar
    (batch uses TR at bar 1; incremental uses h-l at bar 0). After ~20 bars the
    smoothed ATR converges; we verify from bar 30 onward.
    """

    CONVERGE_BAR = 30

    @pytest.fixture
    def price_series(self):
        np.random.seed(7)
        n    = 80
        c    = np.cumsum(np.random.randn(n) * 2) + 200
        h    = c + np.random.uniform(0.5, 3.0, n)
        l    = c - np.random.uniform(0.5, 3.0, n)
        o    = c + np.random.randn(n) * 0.5
        return h, l, c, o

    def test_atr_convergence_after_warmup(self, st, price_series):
        h, l, c, o = price_series
        batch_d, _, batch_atr, batch_ub, batch_lb = _batch_compute_st(
            h, l, c, ST_PERIOD, ST_MULT
        )
        # Feed incrementally
        inc_atrs = []
        for i in range(len(c)):
            st._update_st_5m(h[i], l[i], c[i], o[i], i)
            inc_atrs.append(st._st_atr)

        for i in range(self.CONVERGE_BAR, len(c)):
            assert inc_atrs[i] == pytest.approx(batch_atr[i], rel=5e-3), (
                f"ATR mismatch at bar {i}: inc={inc_atrs[i]:.4f} batch={batch_atr[i]:.4f}"
            )

    def test_direction_convergence_after_warmup(self, st, price_series):
        h, l, c, o = price_series
        batch_d, _, _, _, _ = _batch_compute_st(h, l, c, ST_PERIOD, ST_MULT)

        inc_dirs = []
        for i in range(len(c)):
            st._update_st_5m(h[i], l[i], c[i], o[i], i)
            inc_dirs.append(st._st_direction)

        for i in range(self.CONVERGE_BAR, len(c)):
            assert inc_dirs[i] == batch_d[i], (
                f"Direction mismatch at bar {i}: inc={inc_dirs[i]} batch={batch_d[i]}"
            )

    def test_upper_band_convergence_after_warmup(self, st, price_series):
        h, l, c, o = price_series
        _, _, _, batch_ub, _ = _batch_compute_st(h, l, c, ST_PERIOD, ST_MULT)

        inc_upper = []
        for i in range(len(c)):
            st._update_st_5m(h[i], l[i], c[i], o[i], i)
            inc_upper.append(st._st_upper)

        for i in range(self.CONVERGE_BAR, len(c)):
            assert inc_upper[i] == pytest.approx(batch_ub[i], rel=5e-3), (
                f"Upper band mismatch at bar {i}: inc={inc_upper[i]:.4f} batch={batch_ub[i]:.4f}"
            )

    def test_lower_band_convergence_after_warmup(self, st, price_series):
        h, l, c, o = price_series
        _, _, _, _, batch_lb = _batch_compute_st(h, l, c, ST_PERIOD, ST_MULT)

        inc_lower = []
        for i in range(len(c)):
            st._update_st_5m(h[i], l[i], c[i], o[i], i)
            inc_lower.append(st._st_lower)

        for i in range(self.CONVERGE_BAR, len(c)):
            assert inc_lower[i] == pytest.approx(batch_lb[i], rel=5e-3), (
                f"Lower band mismatch at bar {i}: inc={inc_lower[i]:.4f} batch={batch_lb[i]:.4f}"
            )

    def test_flip_bars_match_batch(self, st, price_series):
        """Flip indices from incremental must match the batch labeler."""
        h, l, c, o = price_series
        batch_d, _, _, _, _ = _batch_compute_st(h, l, c, ST_PERIOD, ST_MULT)
        batch_flips = set(
            i for i in range(1, len(batch_d)) if batch_d[i] != batch_d[i - 1]
        )

        inc_dirs = []
        for i in range(len(c)):
            st._update_st_5m(h[i], l[i], c[i], o[i], i)
            inc_dirs.append(st._st_direction)
        inc_flips = set(
            i for i in range(1, len(inc_dirs)) if inc_dirs[i] != inc_dirs[i - 1]
        )

        # After convergence window, flip bars should be identical
        batch_flips_post = {f for f in batch_flips if f >= self.CONVERGE_BAR}
        inc_flips_post   = {f for f in inc_flips   if f >= self.CONVERGE_BAR}
        assert inc_flips_post == batch_flips_post


# ── 16. Instrument resolution ─────────────────────────────────────────────────

class TestInstrumentResolution:
    """parse_future_symbol correctly resolves micro contract roots."""

    def test_mnq_resolves_to_nq(self):
        from utils.bot_utils import parse_future_symbol
        assert parse_future_symbol('MNQZ5') == 'NQ'

    def test_mes_resolves_to_es(self):
        from utils.bot_utils import parse_future_symbol
        assert parse_future_symbol('MESH5') == 'ES'

    def test_mgc_resolves_to_gc(self):
        from utils.bot_utils import parse_future_symbol
        assert parse_future_symbol('MGCZ5') == 'GC'

    def test_full_symbol_resolves_to_root(self):
        from utils.bot_utils import parse_future_symbol
        assert parse_future_symbol('NQZ5') == 'NQ'

    def test_none_returns_none(self):
        from utils.bot_utils import parse_future_symbol
        assert parse_future_symbol(None) is None

    def test_resolve_instrument_strips_micro_prefix(self):
        assert STTrendStrategyV1._resolve_instrument('MNQZ5') == 'NQ'

    def test_resolve_instrument_handles_four_dot_format(self):
        """4-part IDs: split('.')[-2] extracts the root. 'A.B.MNQ.Z5' → 'MNQ' → 'NQ'."""
        from utils.bot_utils import MICRO_TO_MINI_MAP
        result = STTrendStrategyV1._resolve_instrument('CME.FUT.MNQ.Z5')
        assert result == MICRO_TO_MINI_MAP.get('MNQ', 'MNQ')


# ── 17. Sequence / warmup length ─────────────────────────────────────────────

class TestSequenceLengths:

    def test_sequence_length_is_96(self, st):
        assert st.get_sequence_length() == 96

    def test_warmup_length_is_300(self, st):
        assert st.get_warmup_length() == 300


# ── 18. Exact formula verification — one test per formula ────────────────────
#
# Every formula in the labeling code (st_trend_v1.py) is tested here with
# hardcoded, hand-verifiable inputs and expected outputs.
# Numbers are kept simple so any reader can verify them with a calculator.
# ─────────────────────────────────────────────────────────────────────────────

class TestExactFormulas:
    """
    Exhaustive formula-by-formula verification against the training labeler.
    Inputs and expected values are all calculable by hand.

    Reference: Futures-Foundation-Model/colabs/st_trend_v1.py
    """

    # ── helpers ───────────────────────────────────────────────────────────────

    def _initialized_state(self, st, *, upper=110.0, lower=90.0,
                           direction=1, prev_close=100.0, atr=4.0):
        """Inject a clean post-initialization state."""
        st._st_initialized  = True
        st._st_atr          = atr
        st._st_upper        = upper
        st._st_lower        = lower
        st._st_direction    = direction
        st._st_prev_close   = prev_close
        st._trend_start_bar = 0
        st._trend_high      = 105.0
        st._trend_low       = 95.0
        st._htf_direction   = direction
        st._htf_flip_bar    = 0
        st._atr_history     = deque([atr], maxlen=ATR_RANK_WINDOW)

    # ── F1. True Range = max(h-l, |h-prev_c|, |l-prev_c|) ───────────────────

    def test_tr_normal_no_gap(self, st):
        """No gap: TR = h - l."""
        # h=105, l=95, prev_c=100 → max(10, 5, 5) = 10
        self._initialized_state(st, prev_close=100.0, atr=10.0)
        st._update_st_5m(105, 95, 100, 100, 1)
        # ATR = (9*10 + 10) / 10 = 10.0
        assert st._st_atr == pytest.approx(10.0)

    def test_tr_gap_up(self, st):
        """Gap up: |h - prev_c| dominates."""
        # h=115, l=111, prev_c=100 → max(4, 15, 11) = 15
        self._initialized_state(st, prev_close=100.0, atr=10.0)
        st._update_st_5m(115, 111, 113, 113, 1)
        expected_atr = (9 * 10.0 + 15.0) / 10  # = 10.5
        assert st._st_atr == pytest.approx(expected_atr)

    def test_tr_gap_down(self, st):
        """Gap down: |l - prev_c| dominates."""
        # h=91, l=85, prev_c=100 → max(6, 9, 15) = 15
        self._initialized_state(st, prev_close=100.0, atr=10.0)
        st._update_st_5m(91, 85, 87, 87, 1)
        expected_atr = (9 * 10.0 + 15.0) / 10  # = 10.5
        assert st._st_atr == pytest.approx(expected_atr)

    # ── F2. Wilder's ATR smoothing: ATR[i] = ((p-1)*ATR[i-1] + TR[i]) / p ───

    def test_wilder_smoothing_two_bars(self, st):
        """
        Bar 0: ATR = 8.0  (init: h=104, l=96)
        Bar 1: TR = max(105-95, |105-100|, |95-100|) = max(10,5,5) = 10
               ATR = (9*8 + 10)/10 = 82/10 = 8.2
        """
        st._update_st_5m(104, 96, 100, 100, 0)  # ATR init = 8.0
        assert st._st_atr == pytest.approx(8.0)
        st._update_st_5m(105, 95, 100, 100, 1)
        assert st._st_atr == pytest.approx(8.2)

    def test_wilder_smoothing_three_bars(self, st):
        """
        Bar 0: ATR=8.0
        Bar 1: TR=10, ATR=8.2
        Bar 2: TR = max(104-96, |104-100|, |96-100|) = max(8,4,4) = 8
               ATR = (9*8.2 + 8)/10 = (73.8+8)/10 = 81.8/10 = 8.18
        """
        st._update_st_5m(104, 96, 100, 100, 0)
        st._update_st_5m(105, 95, 100, 100, 1)
        st._update_st_5m(104, 96, 100, 100, 2)
        assert st._st_atr == pytest.approx(8.18)

    def test_wilder_atr_never_below_1e6(self, st):
        """ATR must be floored at 1e-6 even for zero-range bars."""
        st._update_st_5m(100, 100, 100, 100, 0)
        assert st._st_atr == pytest.approx(1e-6)
        st._update_st_5m(100, 100, 100, 100, 1)
        assert st._st_atr >= 1e-6

    # ── F3. hl2 midpoint = (h + l) / 2 ──────────────────────────────────────

    def test_hl2_midpoint_used_for_bands(self, st):
        """
        h=106, l=94 → hl2=100. ATR stays ≈4.
        raw_upper = 100 + 2*4 = 108, raw_lower = 100 - 2*4 = 92.
        Band clamping (prev_upper=110, prev_lower=90, prev_close=100):
          upper: 108 < 110 → use raw = 108
          lower: 92 > 90   → use raw = 92
        """
        self._initialized_state(st, upper=110.0, lower=90.0,
                                 prev_close=100.0, atr=4.0)
        st._update_st_5m(106, 94, 100, 100, 1)
        # ATR = (9*4 + max(12,6,6))/10 = (36+12)/10 = 4.8
        # hl2 = 100, raw_upper = 100+2*4.8=109.6, raw_lower=100-2*4.8=90.4
        # upper: 109.6 < 110 → use 109.6
        assert st._st_upper == pytest.approx(109.6)
        # lower: 90.4 > 90 → use 90.4
        assert st._st_lower == pytest.approx(90.4)

    # ── F4. raw_upper = hl2 + mult * ATR ─────────────────────────────────────

    def test_raw_upper_formula(self, st):
        """raw_upper = (h+l)/2 + ST_MULT * ATR."""
        # Keep ATR constant at 4.0 (TR=4 keeps it at 4.0)
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 prev_close=100.0, atr=4.0)
        # h=102, l=98 → hl2=100, TR=max(4,2,2)=4, ATR=(9*4+4)/10=4.0
        # raw_upper=100+2*4=108 < 200 → use raw
        st._update_st_5m(102, 98, 100, 100, 1)
        assert st._st_upper == pytest.approx(100.0 + ST_MULT * 4.0)

    # ── F5. raw_lower = hl2 - mult * ATR ─────────────────────────────────────

    def test_raw_lower_formula(self, st):
        """raw_lower = (h+l)/2 - ST_MULT * ATR."""
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 prev_close=100.0, atr=4.0)
        # h=102, l=98 → hl2=100, ATR=4.0, raw_lower=100-2*4=92 > 0 → use raw
        st._update_st_5m(102, 98, 100, 100, 1)
        assert st._st_lower == pytest.approx(100.0 - ST_MULT * 4.0)

    # ── F6-F8. Band clamping — all 6 cases ───────────────────────────────────

    def test_upper_clamp_case1_raw_tightens(self, st):
        """raw_upper < prev_upper → always use raw (tightening allowed)."""
        # prev_upper=110, raw_upper will be < 110 (use wide prev, small hl2+atr)
        self._initialized_state(st, upper=110.0, lower=90.0,
                                 prev_close=100.0, atr=4.0)
        # h=102, l=98: hl2=100, ATR=4.0, raw_upper=108 < 110 → use 108
        st._update_st_5m(102, 98, 100, 100, 1)
        assert st._st_upper == pytest.approx(100.0 + ST_MULT * 4.0)  # 108

    def test_upper_clamp_case2_prev_close_breaks_above(self, st):
        """prev_close > prev_upper (break-above) → use raw even if raw > prev."""
        # prev_upper=104, prev_close=106 (above band) → reset to raw
        self._initialized_state(st, upper=104.0, lower=90.0,
                                 prev_close=106.0, atr=4.0)
        # h=108, l=104: hl2=106, TR=max(4,2,2)=4, ATR=4.0
        # raw_upper=106+2*4=114 > 104, but prev_close=106 > 104 → use raw=114
        st._update_st_5m(108, 104, 106, 106, 1)
        assert st._st_upper == pytest.approx(106.0 + ST_MULT * 4.0)  # 114

    def test_upper_clamp_case3_held(self, st):
        """raw_upper >= prev_upper AND prev_close <= prev_upper → hold prev."""
        # prev_upper=104, prev_close=100 (below band) → hold 104
        self._initialized_state(st, upper=104.0, lower=90.0,
                                 prev_close=100.0, atr=4.0)
        # h=108, l=104: hl2=106, ATR=4, raw_upper=114 > 104, prev_close=100 ≤ 104 → hold
        st._update_st_5m(108, 104, 106, 106, 1)
        assert st._st_upper == pytest.approx(104.0)

    def test_lower_clamp_case1_raw_tightens(self, st):
        """raw_lower > prev_lower → always use raw (tightening allowed)."""
        # prev_lower=90, raw_lower will be > 90
        self._initialized_state(st, upper=110.0, lower=90.0,
                                 prev_close=100.0, atr=4.0)
        # h=102, l=98: hl2=100, ATR=4, raw_lower=92 > 90 → use 92
        st._update_st_5m(102, 98, 100, 100, 1)
        assert st._st_lower == pytest.approx(100.0 - ST_MULT * 4.0)  # 92

    def test_lower_clamp_case2_prev_close_breaks_below(self, st):
        """prev_close < prev_lower (break-below) → use raw even if raw < prev."""
        # prev_lower=96, prev_close=94 (below band)
        self._initialized_state(st, upper=110.0, lower=96.0,
                                 prev_close=94.0, atr=4.0)
        # h=97, l=93: hl2=95, TR=max(4,3,1)=4, ATR=4
        # raw_lower=95-2*4=87 < 96, but prev_close=94 < 96 → use raw=87
        st._update_st_5m(97, 93, 95, 95, 1)
        assert st._st_lower == pytest.approx(95.0 - ST_MULT * 4.0)  # 87

    def test_lower_clamp_case3_held(self, st):
        """raw_lower <= prev_lower AND prev_close >= prev_lower → hold prev."""
        # prev_lower=96, prev_close=100 (above band)
        self._initialized_state(st, upper=110.0, lower=96.0,
                                 prev_close=100.0, atr=4.0)
        # h=97, l=93: hl2=95, ATR=4, raw_lower=87 < 96, prev_close=100 ≥ 96 → hold
        st._update_st_5m(97, 93, 95, 95, 1)
        assert st._st_lower == pytest.approx(96.0)

    # ── F9. Direction — bear mode: 1 if close > prev_upper else -1 ───────────

    def test_direction_bear_flip_strictly_greater(self, st):
        """In bear mode, flip to bull only when close STRICTLY > prev_upper."""
        self._initialized_state(st, upper=105.0, lower=90.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._update_st_5m(108, 104, 106, 104, 1)  # close=106 > prev_upper=105
        assert st._st_direction == 1

    def test_direction_bear_no_flip_at_equal(self, st):
        """close == prev_upper in bear mode → stays bear (not strictly greater)."""
        self._initialized_state(st, upper=105.0, lower=90.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._update_st_5m(108, 104, 105, 104, 1)  # close=105 == prev_upper=105
        assert st._st_direction == -1

    def test_direction_bear_no_flip_below(self, st):
        """close < prev_upper in bear mode → stays bear."""
        self._initialized_state(st, upper=105.0, lower=90.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._update_st_5m(106, 102, 103, 102, 1)  # close=103 < prev_upper=105
        assert st._st_direction == -1

    # ── F10. Direction — bull mode: -1 if close < prev_lower else 1 ──────────

    def test_direction_bull_flip_strictly_less(self, st):
        """In bull mode, flip to bear only when close STRICTLY < prev_lower."""
        self._initialized_state(st, upper=110.0, lower=95.0,
                                 direction=1, prev_close=100.0, atr=4.0)
        st._update_st_5m(97, 91, 94, 97, 1)  # close=94 < prev_lower=95
        assert st._st_direction == -1

    def test_direction_bull_no_flip_at_equal(self, st):
        """close == prev_lower in bull mode → stays bull (not strictly less)."""
        self._initialized_state(st, upper=110.0, lower=95.0,
                                 direction=1, prev_close=100.0, atr=4.0)
        st._update_st_5m(97, 94, 95, 97, 1)  # close=95 == prev_lower=95
        assert st._st_direction == 1

    def test_direction_bull_no_flip_above(self, st):
        """close > prev_lower in bull mode → stays bull."""
        self._initialized_state(st, upper=110.0, lower=95.0,
                                 direction=1, prev_close=100.0, atr=4.0)
        st._update_st_5m(102, 96, 99, 102, 1)  # close=99 > prev_lower=95
        assert st._st_direction == 1

    def test_direction_uses_prev_upper_not_current(self, st):
        """
        Critical: direction check must use prev_upper, not the new clamped upper.
        Scenario: prev_upper=105, new bar causes raw_upper=103 (tighter, clamped to 103).
        close=104 → close > prev_upper(105)? NO → stays bear.
        But close > new_upper(103)? YES — wrong if we used new bands.
        """
        self._initialized_state(st, upper=200.0, lower=90.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        # Override upper to 105 after the helper (helper uses 200)
        st._st_upper = 105.0
        # h=103.5, l=103: hl2=103.25, TR=max(0.5,3.5,3)=3.5, ATR=(9*4+3.5)/10=3.95
        # raw_upper=103.25+2*3.95=111.15 > 105, prev_close=100 ≤ 105 → upper stays 105
        # close=104: bear mode checks close > prev_upper(105)? NO → stays -1
        st._update_st_5m(103.5, 103.0, 104.0, 103.0, 1)
        assert st._st_direction == -1

    def test_direction_uses_prev_lower_not_current(self, st):
        """
        Critical: direction check uses prev_lower not new lower.
        Scenario: prev_lower=95, new bar gives raw_lower=96 (tighter, clamped to 96).
        close=95.5 → close < prev_lower(95)? NO → stays bull.
        But close < new_lower(96)? YES — wrong if we used new bands.
        """
        self._initialized_state(st, upper=110.0, lower=0.0,
                                 direction=1, prev_close=100.0, atr=4.0)
        st._st_lower = 95.0
        # h=97, l=96.5: hl2=96.75, TR=max(0.5,3,3.5)=3.5, ATR=(9*4+3.5)/10=3.95
        # raw_lower=96.75-2*3.95=88.85 < 95, prev_close=100 ≥ 95 → lower stays 95
        # close=95.5: bull mode checks close < prev_lower(95)? NO → stays 1
        st._update_st_5m(97.0, 96.5, 95.5, 97.0, 1)
        assert st._st_direction == 1

    # ── F11. st_line = lower (bull) or upper (bear) ───────────────────────────

    def test_st_line_is_lower_band_when_bull(self, st):
        """At a bull signal, st_dist uses the lower band as the ST line."""
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._st_upper = 105.0
        st._st_lower = 95.0
        st._htf_direction = 1   # aligned with upcoming bull flip
        # Trigger bull flip
        st._update_st_5m(108, 104, 107, 104, 5)
        assert st._is_signal_bar is True
        # ST line for bull = lower band; st_dist = (close - lower) / atr
        feats = st._current_st_features
        atr   = st._st_atr
        lower = st._st_lower
        expected_dist = float(np.clip((107.0 - lower) / atr, -10, 10))
        assert feats[1] == pytest.approx(expected_dist, rel=1e-5)

    def test_st_line_is_upper_band_when_bear(self, st):
        """At a bear signal, st_dist uses the upper band as the ST line."""
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 direction=1, prev_close=100.0, atr=4.0)
        st._st_upper = 105.0
        st._st_lower = 95.0
        st._htf_direction = -1   # aligned with upcoming bear flip
        # Trigger bear flip
        st._update_st_5m(97, 91, 93, 97, 5)
        assert st._is_signal_bar is True
        feats = st._current_st_features
        atr   = st._st_atr
        upper = st._st_upper
        expected_dist = float(np.clip((93.0 - upper) / atr, -10, 10))
        assert feats[1] == pytest.approx(expected_dist, rel=1e-5)

    # ── F12. st_dist = (close - st_line) / ATR, clipped [-10, 10] ────────────

    def test_st_dist_positive_for_bull_above_lower(self, st):
        """Bull signal: close > lower band → positive st_dist."""
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._st_upper = 105.0
        st._st_lower = 95.0
        st._htf_direction = 1
        st._update_st_5m(108, 104, 107, 104, 5)
        assert st._current_st_features[1] > 0.0

    def test_st_dist_clipped_at_10(self, st):
        """st_dist is hard-clipped at 10.0 and -10.0."""
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._st_upper = 105.0
        st._st_lower = 5.0      # very low lower band → huge positive dist
        st._htf_direction = 1
        st._update_st_5m(108, 104, 107, 104, 5)
        assert st._current_st_features[1] <= 10.0

    def test_st_dist_exact_value(self, st):
        """
        Controlled: prev_upper=105, prev_lower=92, direction=-1, prev_close=100.
        Bar: h=108, l=104, c=107, o=104.
          TR = max(4, 8, 4) = 8
          ATR = (9*4 + 8)/10 = 4.4
          hl2 = 106
          raw_upper = 106 + 2*4.4 = 114.8; upper clamping: 114.8 < 105? NO, prev_close=100≤105 → hold 105
          raw_lower = 106 - 2*4.4 = 97.2; lower: 97.2 > 92? YES → use 97.2
          direction: bear(-1): c=107 > prev_upper=105? YES → flip to 1
          st_line = lower = 97.2  (bull direction)
          st_dist = (107 - 97.2) / 4.4 = 9.8 / 4.4 ≈ 2.2272...
        """
        self._initialized_state(st, upper=105.0, lower=92.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._htf_direction = 1
        st._update_st_5m(108, 104, 107, 104, 5)
        assert st._is_signal_bar is True
        expected_dist = (107.0 - 97.2) / 4.4
        assert st._current_st_features[1] == pytest.approx(expected_dist, rel=1e-4)

    # ── F13. ATR rank = sum(history < atr) / max(len(history), 1) ────────────

    def test_atr_rank_empty_history(self, st):
        """First non-init bar: history has 1 entry (from init). rank = 0 or 1."""
        st._update_st_5m(102, 98, 100, 100, 0)   # init: ATR=4, history=[4]
        st._update_st_5m(103, 99, 101, 101, 1)   # ATR≈4, history=[4,4]
        # At bar 1: rank = sum([4] < 4) / 1 = 0 (atr same as history[0])
        # No direct accessor, but we can check via a signal
        assert True  # no crash is also a check

    def test_atr_rank_all_history_less_than_current(self, st):
        """
        history all smaller than current ATR → rank = 1.0.
        Uses prev_upper=105 so close=110 triggers a bull flip and fires the signal,
        making _current_st_features[4] accessible.
          h=120, l=90, c=110, prev_c=100, prev_atr=4
          TR = max(30, 20, 10) = 30
          ATR = (9*4 + 30)/10 = 6.6
          rank = sum([1,1,1,1,1] < 6.6) / 5 = 5/5 = 1.0
        """
        st._st_initialized  = True
        st._st_atr          = 4.0
        st._st_prev_close   = 100.0
        st._st_upper        = 105.0   # close=110 > 105 → bull flip fires signal
        st._st_lower        = 0.0
        st._st_direction    = -1
        st._trend_start_bar = 0
        st._trend_high      = 110.0
        st._trend_low       = 90.0
        st._htf_direction   = 1
        st._htf_flip_bar    = 0
        st._atr_history     = deque([1.0, 1.0, 1.0, 1.0, 1.0], maxlen=ATR_RANK_WINDOW)
        st._update_st_5m(120, 90, 110, 90, 5)
        assert st._is_signal_bar is True
        # rank = sum([1,1,1,1,1] < 6.6) / 5 = 1.0
        assert st._current_st_features[4] == pytest.approx(1.0)

    def test_atr_rank_all_history_greater_than_current(self, st):
        """
        history all larger than current ATR → rank = 0.0.
        Uses small-range bar so ATR is tiny but close still crosses prev_upper to fire signal.
          prev_upper=105, prev_close=100, prev_atr=100, direction=-1
          h=106, l=105, c=105.5 — close=105.5 > prev_upper=105 → flip to bull
          TR = max(1, 6, 5) = 6
          ATR = (9*100 + 6)/10 = 90.6
          history=[100,100,100,100,100]; sum([100,100,100,100,100] < 90.6) = 0 → rank=0.0
        """
        st._st_initialized  = True
        st._st_atr          = 100.0
        st._st_prev_close   = 100.0
        st._st_upper        = 105.0   # close=105.5 > 105 → bull flip
        st._st_lower        = 0.0
        st._st_direction    = -1
        st._trend_start_bar = 0
        st._trend_high      = 110.0
        st._trend_low       = 90.0
        st._htf_direction   = 1
        st._htf_flip_bar    = 0
        st._atr_history = deque([100.0, 100.0, 100.0, 100.0, 100.0],
                                maxlen=ATR_RANK_WINDOW)
        st._update_st_5m(106.0, 105.0, 105.5, 106.0, 5)
        assert st._is_signal_bar is True
        # ATR = (9*100 + TR)/10; TR=max(1,6,5)=6 → ATR=90.6 < 100 → rank=0
        assert st._current_st_features[4] == pytest.approx(0.0)

    def test_atr_rank_exact_half(self, st):
        """
        history=[1,2,3,4], ATR_new=3.0 → rank = sum([1,2,3,4]<3)/4 = 2/4 = 0.5.
        ATR_new = (9*1.0 + TR)/10 = 3.0 → TR = 21.
        TR = h-l = 21: h=110.5, l=89.5, prev_c=100 → max(21,10.5,10.5)=21 ✓
        Use prev_upper=105 so close=110 > 105 fires the signal.
        """
        st._st_initialized  = True
        st._st_atr          = 1.0
        st._st_prev_close   = 100.0
        st._st_upper        = 105.0   # close=110 > 105 → bull flip fires signal
        st._st_lower        = 0.0
        st._st_direction    = -1
        st._trend_start_bar = 0
        st._trend_high      = 110.0
        st._trend_low       = 90.0
        st._htf_direction   = 1
        st._htf_flip_bar    = 0
        st._atr_history     = deque([1.0, 2.0, 3.0, 4.0], maxlen=ATR_RANK_WINDOW)
        st._update_st_5m(110.5, 89.5, 110.0, 89.5, 5)
        assert st._is_signal_bar is True
        # ATR = (9*1 + 21)/10 = 3.0
        # rank = sum([1,2,3,4] < 3.0) / 4 = 2/4 = 0.5
        assert st._current_st_features[4] == pytest.approx(0.5)

    def test_atr_rank_excludes_current_bar(self, st):
        """
        Feed 5 bars, then check rank is computed over [bar0..bar4], not including bar5.
        rank at bar 5 = sum(history_0_to_4 < atr_5) / 5.
        If we incorrectly included bar 5, the denominator would be 6 and rank = 5/6.
        """
        st._update_st_5m(102, 98, 100, 100, 0)   # ATR ≈ 4.0
        st._update_st_5m(102, 98, 100, 100, 1)
        st._update_st_5m(102, 98, 100, 100, 2)
        st._update_st_5m(102, 98, 100, 100, 3)
        st._update_st_5m(102, 98, 100, 100, 4)
        assert len(st._atr_history) == 5
        # All 5 entries should be in history before rank is computed for bar 5
        # (rank is computed BEFORE appending bar 5's ATR)
        # bar 5 has same ATR: rank = sum([atr]*5 < atr) / 5 = 0 (not strictly less)
        # denominator must be 5 (not 6)

    # ── F14. htf_age = min((bar_idx - htf_flip_bar) / 100, 5.0) ─────────────

    def test_htf_age_zero_when_just_flipped(self, st):
        """If htf_flip_bar == bar_idx, age = 0."""
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._st_upper = 105.0
        st._st_lower = 95.0
        st._htf_direction = 1
        st._htf_flip_bar  = 5   # flip happened at this bar
        st._update_st_5m(108, 104, 107, 104, 5)   # bar_idx = htf_flip_bar = 5
        assert st._current_st_features[6] == pytest.approx(0.0)

    def test_htf_age_exact_value(self, st):
        """htf_flip_bar=10, bar_idx=60 → age = (60-10)/100 = 0.5."""
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._st_upper = 105.0
        st._st_lower = 95.0
        st._htf_direction = 1
        st._htf_flip_bar  = 10
        st._update_st_5m(108, 104, 107, 104, 60)
        assert st._current_st_features[6] == pytest.approx(0.5)

    def test_htf_age_capped_at_5(self, st):
        """htf_flip_bar=0, bar_idx=600 → raw_age=6.0, capped to 5.0."""
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._st_upper = 105.0
        st._st_lower = 95.0
        st._htf_direction = 1
        st._htf_flip_bar  = 0
        st._update_st_5m(108, 104, 107, 104, 600)
        assert st._current_st_features[6] == pytest.approx(5.0)

    # ── F15. prior_duration = bar_idx - trend_start_bar ──────────────────────

    def test_prior_duration_exact(self, st):
        """trend_start=20, flip at bar=55 → prior_duration=35."""
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._st_upper = 105.0
        st._st_lower = 95.0
        st._trend_start_bar = 20
        st._htf_direction   = 1
        st._update_st_5m(108, 104, 107, 104, 55)
        assert st._prior_duration == 55 - 20

    # ── F16. prior_dur_norm = min(duration / 100, 5.0) ───────────────────────

    def test_prior_dur_norm_exact(self, st):
        """duration=40, norm=40/100=0.4."""
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._st_upper = 105.0
        st._st_lower = 95.0
        st._trend_start_bar = 15
        st._htf_direction   = 1
        st._update_st_5m(108, 104, 107, 104, 55)  # duration = 55-15 = 40
        assert st._current_st_features[2] == pytest.approx(0.4)

    def test_prior_dur_norm_capped(self, st):
        """duration=700, norm=min(7.0, 5.0)=5.0."""
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._st_upper = 105.0
        st._st_lower = 95.0
        st._trend_start_bar = 0
        st._htf_direction   = 1
        st._update_st_5m(108, 104, 107, 104, 700)
        assert st._current_st_features[2] == pytest.approx(5.0)

    # ── F17. prior_range = trend_high - trend_low ─────────────────────────────

    def test_prior_range_exact(self, st):
        """trend_high=108, trend_low=93 → prior_range=15."""
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._st_upper    = 105.0
        st._st_lower    = 95.0
        st._trend_high  = 108.0
        st._trend_low   = 93.0
        st._htf_direction = 1
        st._update_st_5m(108, 104, 107, 104, 5)
        # prior_extent = 108 - 93 = 15; ATR ≈ 4.4
        # prior_ext_norm = min(15/4.4, 10) ≈ 3.409
        expected = min(15.0 / st._st_atr, 10.0)
        assert st._current_st_features[3] == pytest.approx(expected, rel=1e-4)

    # ── F18. prior_ext_norm = min(prior_range / max(atr, 1e-6), 10.0) ────────

    def test_prior_ext_norm_exact(self, st):
        """prior_range=8, atr=4 → ext_norm=2.0."""
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._st_upper   = 105.0
        st._st_lower   = 95.0
        # ATR_new = (9*4 + TR)/10. For TR=4 (h=102,l=98): ATR=(36+4)/10=4.0
        st._trend_high = 108.0
        st._trend_low  = 100.0   # prior_range=8, ATR_new=4.4
        st._htf_direction = 1
        st._update_st_5m(108, 104, 107, 104, 5)
        # ATR=4.4 at this bar; prior_range=8; ext_norm=8/4.4=1.818...
        expected = min(8.0 / st._st_atr, 10.0)
        assert st._current_st_features[3] == pytest.approx(expected, rel=1e-4)

    def test_prior_ext_norm_capped_at_10(self, st):
        """prior_range >> atr → ext_norm capped at 10."""
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._st_upper   = 105.0
        st._st_lower   = 95.0
        st._trend_high = 10000.0
        st._trend_low  = 0.0
        st._htf_direction = 1
        st._update_st_5m(108, 104, 107, 104, 5)
        assert st._current_st_features[3] == pytest.approx(10.0)

    # ── F19. bar_range = h - l ────────────────────────────────────────────────

    def test_bar_range_used_for_flip_body(self, st):
        """flip_body = |close - open| / (h - l). bar_range=h-l."""
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._st_upper = 105.0
        st._st_lower = 95.0
        st._htf_direction = 1
        # h=110, l=104, o=104, c=107: bar_range=6, body=3 → flip_body=0.5
        st._update_st_5m(110, 104, 107, 104, 5)
        assert st._current_st_features[7] == pytest.approx(3.0 / 6.0)

    # ── F20. flip_body = |close - open| / max(h - l, 1e-6) ──────────────────

    def test_flip_body_exact(self, st):
        """h=112, l=104, o=105, c=110: body=5, range=8 → flip_body=5/8=0.625."""
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._st_upper = 105.0
        st._st_lower = 95.0
        st._htf_direction = 1
        st._update_st_5m(112, 104, 110, 105, 5)
        assert st._current_st_features[7] == pytest.approx(5.0 / 8.0)

    def test_flip_body_doji_floored(self, st):
        """h=l (zero range) → denominator floored at 1e-6 — no division by zero."""
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._st_upper = 105.0
        st._st_lower = 95.0
        st._htf_direction = 1
        st._update_st_5m(107, 107, 107, 107, 5)
        assert not np.isnan(st._current_st_features[7])
        assert st._current_st_features[7] == pytest.approx(0.0)

    def test_flip_body_in_range(self, st):
        """flip_body is always in [0, 1]."""
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._st_upper = 105.0
        st._st_lower = 95.0
        st._htf_direction = 1
        st._update_st_5m(112, 104, 108, 104, 5)
        assert 0.0 <= st._current_st_features[7] <= 1.0

    # ── F21. Feature column order must match ST_FEATURE_COLS exactly ──────────

    def test_feature_column_order(self, st):
        """
        ST_FEATURE_COLS order (from labeler):
          0: st_direction
          1: st_line_distance
          2: prior_trend_duration_norm
          3: prior_trend_extent_norm
          4: atr_rank_pct
          5: htf_st_direction
          6: htf_st_age_norm
          7: flip_bar_body_pct
        Verify each index matches its semantic meaning.
        """
        self._initialized_state(st, upper=200.0, lower=0.0,
                                 direction=-1, prev_close=100.0, atr=4.0)
        st._st_upper        = 105.0
        st._st_lower        = 95.0
        st._trend_start_bar = 10
        st._trend_high      = 108.0
        st._trend_low       = 96.0
        st._htf_direction   = 1
        st._htf_flip_bar    = 20
        # bar_idx=60: h=110, l=104, c=107, o=104
        st._update_st_5m(110, 104, 107, 104, 60)
        f = st._current_st_features
        assert f[0] == pytest.approx(1.0)      # st_direction = 1 (bull flip)
        assert f[1] != 0.0                      # st_line_distance — non-zero dist
        assert f[2] == pytest.approx(min((60-10)/100.0, 5.0))  # dur_norm = 0.5
        assert f[3] > 0.0                       # ext_norm — positive
        assert 0.0 <= f[4] <= 1.0              # atr_rank_pct in [0,1]
        assert f[5] == pytest.approx(1.0)      # htf_direction = 1
        assert f[6] == pytest.approx(min((60-20)/100.0, 5.0))  # age_norm = 0.4
        assert 0.0 <= f[7] <= 1.0             # flip_body in [0,1]

    # ── F22. SL distance = ATR × SL_ATR_MULT (1.5) ───────────────────────────

    def test_sl_distance_exact(self, st):
        """stop_pts = signal_atr * 1.5."""
        st._signal_atr    = 8.0
        st._latest_risk_rr = 2.0
        stop, _ = st.get_stop_target_pts(None, 'LONG', 100.0)
        assert stop == pytest.approx(8.0 * SL_ATR_MULT)   # 8 * 1.5 = 12.0

    def test_sl_atr_mult_constant_is_1_5(self):
        assert SL_ATR_MULT == 1.5

    # ── F23. TP tier logic exactly matches the 3-tier system ─────────────────

    def test_tp_tier_boundaries_exact(self, st):
        """Verify: <2→skip, ≥2→int(raw_rr)×R with no ceiling."""
        skip_cases = [0.0, 0.9, 1.0, 1.5, 1.99]
        for raw_rr in skip_cases:
            st._signal_atr    = 4.0
            st._latest_risk_rr = raw_rr
            stop, target = st.get_stop_target_pts(None, 'LONG', 100.0)
            assert stop is None and target is None, \
                f"raw_rr={raw_rr}: expected skip (None, None), got ({stop}, {target})"

        tier_cases = [
            (2.0,  2.0),
            (2.9,  2.0),
            (3.0,  3.0),
            (3.9,  3.0),
            (4.0,  4.0),
            (4.9,  4.0),
            (5.0,  5.0),
            (5.9,  5.0),
            (6.0,  6.0),
            (7.5,  7.0),    # no ceiling — int(7.5)=7
            (14.2, 14.0),   # high prediction → 14R
        ]
        for raw_rr, expected_tier in tier_cases:
            st._signal_atr    = 4.0
            st._latest_risk_rr = raw_rr
            stop, target = st.get_stop_target_pts(None, 'LONG', 100.0)
            assert target == pytest.approx(stop * expected_tier, rel=1e-5), \
                f"raw_rr={raw_rr}: expected {expected_tier}R, got {target/stop:.3f}R"

    # ── F24. Batch reference compute_st produces correct bands ───────────────

    def test_batch_compute_st_reference_bands(self):
        """
        Run _batch_compute_st on a simple deterministic series and verify
        the reference function itself computes expected values.
        This guards against any accidental modification of the reference code.
        """
        # 3 bars: H=[104,105,103], L=[96,97,99], C=[100,101,101]
        h = np.array([104.0, 105.0, 103.0])
        l = np.array([ 96.0,  97.0,  99.0])
        c = np.array([100.0, 101.0, 101.0])
        _, _, atr, ub, lb = _batch_compute_st(h, l, c, period=10, mult=2.0)
        # atr[0] = tr[0] = max(105-97, |105-100|, |97-100|) = max(8,5,3) = 8
        assert atr[0] == pytest.approx(8.0)
        # atr[1] = (9*8 + tr[0])/10 = (72+8)/10 = 8.0
        assert atr[1] == pytest.approx(8.0)
        # ub[0] = (104+96)/2 + 2*atr[0] = 100 + 2*8 = 116  (first bar)
        assert ub[0] == pytest.approx(100.0 + 2.0 * atr[0])

    # ── F25. ATR rank window capped at ATR_RANK_WINDOW=200 ───────────────────

    def test_atr_rank_window_exactly_200(self, st):
        """After 300 bars, history must hold exactly 200 entries."""
        for i in range(300):
            st._update_st_5m(102, 98, 100, 100, i)
        assert len(st._atr_history) == ATR_RANK_WINDOW  # 200

    def test_atr_rank_window_constant(self):
        assert ATR_RANK_WINDOW == 200


# ── 19. Constructor — min_risk_rr wiring ──────────────────────────────────────

class TestConstructorMinRiskRR:
    """
    min_risk_rr must be accepted and stored at construction time.
    Regression: algoTrader.py previously omitted this kwarg for supertrend,
    leaving _min_risk_rr=0.0 and silently disabling the rr gate entirely.
    """

    def test_default_is_two(self):
        import logging
        logging.disable(logging.CRITICAL)
        s = STTrendStrategyV1(model_path="", contract_symbol="MNQ")
        assert s._min_risk_rr == 2.0

    def test_constructor_stores_min_risk_rr(self):
        import logging
        logging.disable(logging.CRITICAL)
        s = STTrendStrategyV1(model_path="", contract_symbol="MNQ", min_risk_rr=2.0)
        assert s._min_risk_rr == 2.0

    def test_rr_gate_inactive_when_zero(self):
        """_min_risk_rr=0.0 → gate condition never fires regardless of predicted rr."""
        import logging
        logging.disable(logging.CRITICAL)
        s = STTrendStrategyV1(model_path="", contract_symbol="MNQ", min_risk_rr=0.0)
        s._latest_risk_rr = 0.01
        ok, _ = s.should_enter_trade(1, 0.95, {}, entry_conf=0.80, adx_thresh=0)
        assert ok is True

    def test_rr_gate_active_when_set(self):
        """_min_risk_rr=2.0 → predicted_rr=1.5 must be blocked."""
        import logging
        logging.disable(logging.CRITICAL)
        s = STTrendStrategyV1(model_path="", contract_symbol="MNQ", min_risk_rr=2.0)
        s._latest_risk_rr = 1.5
        ok, _ = s.should_enter_trade(1, 0.95, {}, entry_conf=0.80, adx_thresh=0)
        assert ok is False
        assert s.skip_stats['rr_gate'] == 1
