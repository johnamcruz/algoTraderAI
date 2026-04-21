"""Tests for _run_ai_prediction decision logic.

Covers the gates that fire between signal generation and order placement:
  - Existing position blocks duplicate entry (restart-while-in-position)
  - High-confidence multiplier extends target, not stop
  - Stop too tight (< 4 ticks) blocks entry
  - Zero contract size blocks entry
  - No stop/target available blocks entry
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from tests.helpers import ConcreteBot, MockStrategy


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_bot(strategy=None, **kwargs):
    if strategy is None:
        strategy = MockStrategy()
    defaults = dict(
        contract="MNQ",
        size=1,
        timeframe_minutes=5,
        strategy=strategy,
        entry_conf=0.70,
        adx_thresh=0,
        stop_pts=10.0,
        target_pts=20.0,
        risk_amount=50.0,
    )
    defaults.update(kwargs)
    bot = ConcreteBot(**defaults)
    bot._has_existing_position = AsyncMock(return_value=False)
    bot._place_order = AsyncMock()
    return bot


def _fill_bars(bot, n=200):
    """Populate historical_bars with n minimal in-session bars."""
    ts = datetime(2026, 4, 21, 12, 0, 0, tzinfo=timezone.utc).isoformat()
    for _ in range(n):
        bot.historical_bars.append({
            "timestamp": ts,
            "open": 100.0, "high": 101.0,
            "low": 99.0,  "close": 100.5,
            "volume": 1000,
        })


# ── Existing position blocks entry ────────────────────────────────────────────

class TestExistingPositionGate:
    """Mirrors the restart-while-in-position scenario verified live on 2026-04-21."""

    def test_existing_position_suppresses_order(self):
        bot = _make_bot()
        _fill_bars(bot)
        bot._has_existing_position = AsyncMock(return_value=True)

        run(bot._run_ai_prediction())

        bot._place_order.assert_not_called()

    def test_no_position_allows_order(self):
        bot = _make_bot()
        _fill_bars(bot)
        bot._has_existing_position = AsyncMock(return_value=False)

        run(bot._run_ai_prediction())

        bot._place_order.assert_called_once()

    def test_short_signal_also_blocked_when_in_position(self):
        strategy = MockStrategy()
        strategy.predict = MagicMock(return_value=(2, 0.85))
        strategy.should_enter_trade = MagicMock(return_value=(True, "SHORT"))
        bot = _make_bot(strategy=strategy)
        _fill_bars(bot)
        bot._has_existing_position = AsyncMock(return_value=True)

        run(bot._run_ai_prediction())

        bot._place_order.assert_not_called()


# ── High-confidence target multiplier ─────────────────────────────────────────

class TestHighConfMultiplier:
    """At confidence >= 0.90 with multiplier=2.0, take-profit ticks double."""

    # ConcreteBot: tick_size=0.25, tick_value=$0.50
    # stop_pts=10 → stop_ticks=40; target_pts=20 → tp_ticks=80; ×2 → 160

    def test_target_doubled_at_high_confidence(self):
        bot = _make_bot(high_conf_multiplier=2.0, stop_pts=10.0, target_pts=20.0)
        _fill_bars(bot)
        bot.strategy.predict = MagicMock(return_value=(1, 0.92))

        captured = {}
        async def capture(**kwargs):
            captured.update(kwargs)
        bot._place_order = capture

        run(bot._run_ai_prediction())

        assert captured["take_profit_ticks"] == 160  # 20 * 2 / 0.25

    def test_target_unchanged_below_threshold(self):
        bot = _make_bot(high_conf_multiplier=2.0, stop_pts=10.0, target_pts=20.0)
        _fill_bars(bot)
        bot.strategy.predict = MagicMock(return_value=(1, 0.85))  # < 0.90

        captured = {}
        async def capture(**kwargs):
            captured.update(kwargs)
        bot._place_order = capture

        run(bot._run_ai_prediction())

        assert captured["take_profit_ticks"] == 80  # 20 / 0.25, no multiplier

    def test_stop_unchanged_regardless_of_confidence(self):
        """High confidence only extends target — stop risk is never increased."""
        bot = _make_bot(high_conf_multiplier=2.0, stop_pts=10.0, target_pts=20.0)
        _fill_bars(bot)
        bot.strategy.predict = MagicMock(return_value=(1, 0.95))

        captured = {}
        async def capture(**kwargs):
            captured.update(kwargs)
        bot._place_order = capture

        run(bot._run_ai_prediction())

        # stop_ticks = -int(10.0 / 0.25) = -40, unchanged
        assert captured["stop_ticks"] == -40

    def test_multiplier_of_one_is_noop(self):
        bot = _make_bot(high_conf_multiplier=1.0, stop_pts=10.0, target_pts=20.0)
        _fill_bars(bot)
        bot.strategy.predict = MagicMock(return_value=(1, 0.95))

        captured = {}
        async def capture(**kwargs):
            captured.update(kwargs)
        bot._place_order = capture

        run(bot._run_ai_prediction())

        assert captured["take_profit_ticks"] == 80  # unchanged


# ── Minimum stop gate ────────────────────────────────────────────────────────

class TestMinStopGate:
    """Signals with a stop below --min_stop_pts are skipped.

    ConcreteBot: tick_size=0.25
    Default min_stop_pts=1.0 → 4 ticks minimum.
    """

    def test_stop_below_default_minimum_skips_order(self):
        # 0.5 pts / 0.25 tick = 2 ticks < 4 ticks (1.0 pt default)
        bot = _make_bot(stop_pts=0.5, target_pts=5.0)
        _fill_bars(bot)

        run(bot._run_ai_prediction())

        bot._place_order.assert_not_called()

    def test_stop_at_exactly_default_minimum_allows_order(self):
        # 1.0 pts / 0.25 tick = 4 ticks == 4-tick minimum
        bot = _make_bot(stop_pts=1.0, target_pts=5.0)
        _fill_bars(bot)

        run(bot._run_ai_prediction())

        bot._place_order.assert_called_once()

    def test_stop_above_default_minimum_allows_order(self):
        bot = _make_bot(stop_pts=10.0, target_pts=20.0)
        _fill_bars(bot)

        run(bot._run_ai_prediction())

        bot._place_order.assert_called_once()

    def test_custom_min_stop_pts_blocks_previously_valid_stop(self):
        # Raise minimum to 2.0 pts (8 ticks). A 1.5 pt stop that previously
        # passed (6 ticks > 4) is now rejected.
        bot = _make_bot(stop_pts=1.5, target_pts=10.0, min_stop_pts=2.0)
        _fill_bars(bot)

        run(bot._run_ai_prediction())

        bot._place_order.assert_not_called()

    def test_custom_min_stop_pts_allows_stop_at_threshold(self):
        # Exactly at the raised minimum: 2.0 pts / 0.25 tick = 8 ticks
        bot = _make_bot(stop_pts=2.0, target_pts=10.0, min_stop_pts=2.0)
        _fill_bars(bot)

        run(bot._run_ai_prediction())

        bot._place_order.assert_called_once()

    def test_custom_min_stop_pts_allows_stop_above_threshold(self):
        # 4.0 pts > 2.0 pt minimum — should pass
        bot = _make_bot(stop_pts=4.0, target_pts=10.0, min_stop_pts=2.0)
        _fill_bars(bot)

        run(bot._run_ai_prediction())

        bot._place_order.assert_called_once()

    def test_min_stop_pts_zero_disables_gate(self):
        # Setting min_stop_pts=0 means any stop width passes
        bot = _make_bot(stop_pts=0.25, target_pts=5.0, min_stop_pts=0.0)
        _fill_bars(bot)

        run(bot._run_ai_prediction())

        bot._place_order.assert_called_once()


# ── Zero size gate ────────────────────────────────────────────────────────────

class TestZeroSizeGate:
    """When risk budget can't cover even 1 contract at the given stop, skip."""

    def test_oversized_stop_results_in_zero_contracts(self):
        # risk=$1, stop=100pts → 400 ticks × $0.50 = $200/contract >> $1 budget
        bot = _make_bot(risk_amount=1.0, stop_pts=100.0, target_pts=200.0)
        _fill_bars(bot)

        run(bot._run_ai_prediction())

        bot._place_order.assert_not_called()


# ── No stop / target available ────────────────────────────────────────────────

class TestNoStopTargetGate:
    """If neither strategy nor bot config provides stop/target, skip the trade."""

    def test_strategy_and_bot_both_have_no_stop(self):
        strategy = MockStrategy()
        strategy.get_stop_target_pts = MagicMock(return_value=(None, None))
        bot = _make_bot(strategy=strategy, stop_pts=None, target_pts=None)
        _fill_bars(bot)

        run(bot._run_ai_prediction())

        bot._place_order.assert_not_called()

    def test_strategy_stop_overrides_missing_bot_stop(self):
        """Strategy providing stop/target is sufficient even if bot has none."""
        strategy = MockStrategy()
        strategy.get_stop_target_pts = MagicMock(return_value=(10.0, 20.0))
        bot = _make_bot(strategy=strategy, stop_pts=None, target_pts=None)
        _fill_bars(bot)

        run(bot._run_ai_prediction())

        bot._place_order.assert_called_once()

    def test_bot_stop_used_when_strategy_returns_none(self):
        """Bot-level stop_pts/target_pts are the fallback when strategy returns None."""
        strategy = MockStrategy()
        strategy.get_stop_target_pts = MagicMock(return_value=(None, None))
        bot = _make_bot(strategy=strategy, stop_pts=10.0, target_pts=20.0)
        _fill_bars(bot)

        run(bot._run_ai_prediction())

        bot._place_order.assert_called_once()
