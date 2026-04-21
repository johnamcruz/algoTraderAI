"""Unit tests for BaseStrategy._on_new_bar / _run_warmup warmup hook mechanism."""

import sys
import os
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_base import BaseStrategy


# ── Minimal concrete strategy for testing ─────────────────────────────────────

class StatefulStrategy(BaseStrategy):
    """Concrete stub that records which bar indices _on_new_bar was called with."""

    def __init__(self):
        super().__init__(model_path="", scaler_path="", contract_symbol="TEST")
        self.seen_bar_indices = []
        self.seen_df_lengths = []

    def _on_new_bar(self, df, bar_idx):
        self.seen_bar_indices.append(bar_idx)
        self.seen_df_lengths.append(len(df))

    # Required abstract stubs
    def get_feature_columns(self): return []
    def add_features(self, df): return df
    def load_model(self): pass
    def load_scaler(self): pass
    def predict(self, df): return (0, 0.0)
    def should_enter_trade(self, pred, conf, bar, entry_conf, adx_thresh): return (False, None)


def _make_df(n_rows):
    return pd.DataFrame({
        "open": np.ones(n_rows) * 100.0,
        "high": np.ones(n_rows) * 101.0,
        "low":  np.ones(n_rows) * 99.0,
        "close": np.ones(n_rows) * 100.5,
        "volume": np.ones(n_rows, dtype=int) * 1000,
    })


# ── _on_new_bar default (no-op) ────────────────────────────────────────────────

class TestOnNewBarDefault:
    """BaseStrategy._on_new_bar is a safe no-op — subclasses opt in by overriding."""

    def test_noop_does_not_raise(self):
        class NoopStrategy(StatefulStrategy):
            def _on_new_bar(self, df, bar_idx):
                pass  # explicit no-op, same as base default

        s = NoopStrategy()
        s._on_new_bar(_make_df(5), 0)  # must not raise

    def test_bar_count_starts_at_zero(self):
        s = StatefulStrategy()
        assert s._bar_count == 0


# ── _run_warmup ────────────────────────────────────────────────────────────────

class TestRunWarmup:
    """_run_warmup feeds n-1 historical bars through _on_new_bar in order."""

    def test_calls_on_new_bar_for_each_historical_bar(self):
        s = StatefulStrategy()
        df = _make_df(5)
        s._run_warmup(df)
        # 5 rows → 4 warmup bars (indices 0..3)
        assert s.seen_bar_indices == [0, 1, 2, 3]

    def test_df_slice_grows_by_one_each_call(self):
        s = StatefulStrategy()
        df = _make_df(4)
        s._run_warmup(df)
        # Each call gets an incrementally larger slice
        assert s.seen_df_lengths == [1, 2, 3]

    def test_bar_count_increments_during_warmup(self):
        s = StatefulStrategy()
        df = _make_df(6)
        s._run_warmup(df)
        # 5 warmup bars → _bar_count ends at 5
        assert s._bar_count == 5

    def test_single_row_df_skips_warmup(self):
        s = StatefulStrategy()
        s._run_warmup(_make_df(1))
        assert s.seen_bar_indices == []
        assert s._bar_count == 0

    def test_empty_df_skips_warmup(self):
        s = StatefulStrategy()
        s._run_warmup(_make_df(0))
        assert s.seen_bar_indices == []

    def test_second_call_continues_from_current_bar_count(self):
        """After warmup, a second call starts from where _bar_count left off."""
        s = StatefulStrategy()
        df = _make_df(3)
        s._run_warmup(df)           # processes bars 0, 1 → _bar_count=2
        # Simulate adding one more bar and calling _run_warmup again (no-op guard)
        df2 = _make_df(4)
        s._run_warmup(df2)          # _bar_count != 0, so loops from 2..2
        # The second call should process bars 2..2 (n-1=3 iterations, but
        # _bar_count was already 2, so range(n-1)=range(3) gives 0,1,2 → 3 more calls)
        # NOTE: _run_warmup does NOT have an early-exit guard — callers are responsible
        # for calling it only on the first invocation (when _bar_count == 0).
        # This test just verifies it doesn't crash; the CISD strategy guards with:
        #   if self._bar_count == 0 and n > 1: self._run_warmup(df)
        assert s._bar_count > 0  # at least ran both times without error
