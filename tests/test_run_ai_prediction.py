"""Tests for _run_ai_prediction decision logic.

Covers the gates that fire between signal generation and order placement:
  - Existing position blocks duplicate entry (restart-while-in-position)
  - In-memory position guard fires before broker API call (double-entry fix)
  - in_position flag resets on every failure path so the next bar can trade
  - High-confidence multiplier extends target, not stop
  - Stop too tight (< 4 ticks) blocks entry
  - Zero contract size blocks entry
  - No stop/target available blocks entry
  - SimulationBot skips cross-process mutex (no 'account' AttributeError in backtest)
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


def _fill_bars(bot, n=200, vty_atr_14=None):
    """Populate historical_bars with n minimal in-session bars.

    Pass vty_atr_14 (normalised ATR = atr/close) to simulate ATR-based gate tests.
    With close=100.5, vty_atr_14=0.05 → ATR ≈ 5.025 pts.
    """
    ts = datetime(2026, 4, 21, 12, 0, 0, tzinfo=timezone.utc).isoformat()
    bar = {
        "timestamp": ts,
        "open": 100.0, "high": 101.0,
        "low": 99.0,  "close": 100.5,
        "volume": 1000,
    }
    if vty_atr_14 is not None:
        bar["vty_atr_14"] = vty_atr_14
    for _ in range(n):
        bot.historical_bars.append(bar.copy())


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


# ── Dynamic ATR-based minimum stop ───────────────────────────────────────────

class TestDynamicMinStopGate:
    """--min_stop_atr sets a floor relative to ATR14; effective min = max(fixed, atr×mult)."""

    # close=100.5, vty_atr_14=0.05 → ATR ≈ 5.025 pts

    def test_atr_floor_blocks_tight_stop(self):
        # ATR≈5.025 * 0.5 = 2.51 pt floor. stop=2.0 < 2.51 → blocked.
        bot = _make_bot(stop_pts=2.0, target_pts=10.0, min_stop_atr_mult=0.5)
        _fill_bars(bot, vty_atr_14=0.05)

        run(bot._run_ai_prediction())

        bot._place_order.assert_not_called()

    def test_atr_floor_allows_stop_above_threshold(self):
        # ATR≈5.025 * 0.5 = 2.51 pt floor. stop=3.0 > 2.51 → allowed.
        bot = _make_bot(stop_pts=3.0, target_pts=10.0, min_stop_atr_mult=0.5)
        _fill_bars(bot, vty_atr_14=0.05)

        run(bot._run_ai_prediction())

        bot._place_order.assert_called_once()

    def test_fixed_floor_wins_when_higher_than_atr_floor(self):
        # ATR≈5.025 * 0.1 = 0.50 pt. Fixed min_stop_pts=2.0 > 0.50 → fixed floor applies.
        # stop=1.5 < 2.0 → blocked.
        bot = _make_bot(stop_pts=1.5, target_pts=10.0, min_stop_pts=2.0, min_stop_atr_mult=0.1)
        _fill_bars(bot, vty_atr_14=0.05)

        run(bot._run_ai_prediction())

        bot._place_order.assert_not_called()

    def test_atr_floor_wins_when_higher_than_fixed_floor(self):
        # ATR≈5.025 * 0.5 = 2.51 pt. Fixed min_stop_pts=1.0. ATR floor wins.
        # stop=2.0 < 2.51 → blocked.
        bot = _make_bot(stop_pts=2.0, target_pts=10.0, min_stop_pts=1.0, min_stop_atr_mult=0.5)
        _fill_bars(bot, vty_atr_14=0.05)

        run(bot._run_ai_prediction())

        bot._place_order.assert_not_called()

    def test_zero_mult_disables_atr_gate(self):
        # min_stop_atr_mult=0 → ATR gate inactive; only fixed floor (1.0 pt default).
        # stop=1.5 > 1.0 → allowed.
        bot = _make_bot(stop_pts=1.5, target_pts=10.0, min_stop_atr_mult=0.0)
        _fill_bars(bot, vty_atr_14=0.05)

        run(bot._run_ai_prediction())

        bot._place_order.assert_called_once()

    def test_missing_atr_feature_falls_back_to_fixed_floor(self):
        # No vty_atr_14 in bars → atr_pts=0 → ATR floor=0 → fixed floor=1.0 applies.
        # stop=2.0 > 1.0 → allowed.
        bot = _make_bot(stop_pts=2.0, target_pts=10.0, min_stop_atr_mult=0.5)
        _fill_bars(bot)  # no vty_atr_14

        run(bot._run_ai_prediction())

        bot._place_order.assert_called_once()


# ── Order rejection detection ─────────────────────────────────────────────────

class TestOrderRejectionDetection:
    """When _place_order returns falsy (broker rejected), no exception is raised
    and the signal is treated as not placed."""

    def test_rejected_order_does_not_raise(self):
        bot = _make_bot(stop_pts=10.0, target_pts=20.0)
        _fill_bars(bot)
        bot._place_order = AsyncMock(return_value=None)

        run(bot._run_ai_prediction())  # must not raise

        bot._place_order.assert_called_once()

    def test_successful_order_returns_truthy(self):
        bot = _make_bot(stop_pts=10.0, target_pts=20.0)
        _fill_bars(bot)
        bot._place_order = AsyncMock(return_value="order-123")

        run(bot._run_ai_prediction())

        bot._place_order.assert_called_once()

    def test_false_return_also_handled_gracefully(self):
        bot = _make_bot(stop_pts=10.0, target_pts=20.0)
        _fill_bars(bot)
        bot._place_order = AsyncMock(return_value=False)

        run(bot._run_ai_prediction())

        bot._place_order.assert_called_once()


# ── Double-entry guard (bug fix: 2026-04-28) ─────────────────────────────────

class TestInMemoryPositionGuard:
    """in_position=True blocks entry before any broker API call.

    Root cause: two concurrent run2.py processes both passed Position/searchOpen
    before either order was confirmed, resulting in duplicate orders at the same
    bar. Fix: check self.in_position immediately on signal, before awaiting
    _has_existing_position().
    """

    def test_in_memory_flag_blocks_before_broker_check(self):
        """When in_position is already True, _has_existing_position is never called."""
        bot = _make_bot(stop_pts=10.0, target_pts=20.0)
        _fill_bars(bot)
        bot.in_position = True

        run(bot._run_ai_prediction())

        bot._has_existing_position.assert_not_called()
        bot._place_order.assert_not_called()

    def test_in_position_false_proceeds_to_broker_check(self):
        """When in_position is False, the broker API is consulted as normal."""
        bot = _make_bot(stop_pts=10.0, target_pts=20.0)
        _fill_bars(bot)
        bot.in_position = False

        run(bot._run_ai_prediction())

        bot._has_existing_position.assert_called_once()

    def test_in_position_set_before_place_order_is_awaited(self):
        """in_position must be True at the moment _place_order is called.

        Ensures that a second concurrent signal sees the flag and bails
        even if the first _place_order hasn't returned yet.
        """
        bot = _make_bot(stop_pts=10.0, target_pts=20.0)
        _fill_bars(bot)

        observed = {}

        async def capture_state(**kwargs):
            observed["in_position_during_place"] = bot.in_position
            return "order-123"

        bot._place_order = capture_state

        run(bot._run_ai_prediction())

        assert observed.get("in_position_during_place") is True


class TestInPositionReset:
    """in_position must be reset to False on every non-entry path.

    If in_position is left True after a skipped or failed order, the bot
    will refuse all future entries until restart.
    """

    def test_reset_when_order_rejected_by_broker_long(self):
        bot = _make_bot(stop_pts=10.0, target_pts=20.0)
        _fill_bars(bot)
        bot._place_order = AsyncMock(return_value=None)

        run(bot._run_ai_prediction())

        assert bot.in_position is False

    def test_reset_when_order_rejected_by_broker_short(self):
        strategy = MockStrategy()
        strategy.predict = MagicMock(return_value=(2, 0.85))
        strategy.should_enter_trade = MagicMock(return_value=(True, "SHORT"))
        bot = _make_bot(strategy=strategy, stop_pts=10.0, target_pts=20.0)
        _fill_bars(bot)
        bot._place_order = AsyncMock(return_value=None)

        run(bot._run_ai_prediction())

        assert bot.in_position is False

    def test_reset_when_stop_too_tight(self):
        # 0.5 pt stop < 1.0 pt minimum → skipped before _place_order
        bot = _make_bot(stop_pts=0.5, target_pts=5.0)
        _fill_bars(bot)

        run(bot._run_ai_prediction())

        assert bot.in_position is False
        bot._place_order.assert_not_called()

    def test_reset_when_size_is_zero_long(self):
        # risk=$1, stop=100 pts → 0 contracts
        bot = _make_bot(risk_amount=1.0, stop_pts=100.0, target_pts=200.0)
        _fill_bars(bot)

        run(bot._run_ai_prediction())

        assert bot.in_position is False
        bot._place_order.assert_not_called()

    def test_reset_when_size_is_zero_short(self):
        strategy = MockStrategy()
        strategy.predict = MagicMock(return_value=(2, 0.85))
        strategy.should_enter_trade = MagicMock(return_value=(True, "SHORT"))
        bot = _make_bot(strategy=strategy, risk_amount=1.0, stop_pts=100.0, target_pts=200.0)
        _fill_bars(bot)

        run(bot._run_ai_prediction())

        assert bot.in_position is False
        bot._place_order.assert_not_called()

    def test_reset_when_no_stop_target(self):
        strategy = MockStrategy()
        strategy.get_stop_target_pts = MagicMock(return_value=(None, None))
        bot = _make_bot(strategy=strategy, stop_pts=None, target_pts=None)
        _fill_bars(bot)

        run(bot._run_ai_prediction())

        assert bot.in_position is False
        bot._place_order.assert_not_called()

    def test_reset_on_exception_during_prediction(self):
        strategy = MockStrategy()
        strategy.predict = MagicMock(side_effect=RuntimeError("model exploded"))
        bot = _make_bot(strategy=strategy)
        _fill_bars(bot)
        bot.in_position = False

        run(bot._run_ai_prediction())

        assert bot.in_position is False

    def test_successful_order_leaves_in_position_true(self):
        """Sanity check: a successful order must NOT reset in_position."""
        bot = _make_bot(stop_pts=10.0, target_pts=20.0)
        _fill_bars(bot)
        bot._place_order = AsyncMock(return_value="order-123")

        run(bot._run_ai_prediction())

        # _place_order itself sets in_position=True; we just confirm it's still True
        assert bot.in_position is True


class TestCrossProcessOrderLock:
    """Filesystem mutex: prevents two processes from both passing the position
    check and placing orders before either has a broker-confirmed position."""

    def test_lock_blocks_second_process(self):
        """If the lockdir already exists, the signal is skipped — no order placed."""
        import os, tempfile
        bot = _make_bot(stop_pts=10.0, target_pts=20.0)
        bot._is_simulation = False
        _fill_bars(bot)
        bot._place_order = AsyncMock(return_value="order-123")

        lock_dir = os.path.join(tempfile.gettempdir(), f"algo_order_{bot.account}.lockdir")
        os.makedirs(lock_dir, exist_ok=True)
        try:
            run(bot._run_ai_prediction())
            bot._place_order.assert_not_called()
            assert bot.in_position is False
        finally:
            os.rmdir(lock_dir)

    def test_lock_released_after_successful_order(self):
        """Lock must be gone after a successful order so the next signal can trade."""
        import os, tempfile
        bot = _make_bot(stop_pts=10.0, target_pts=20.0)
        bot._is_simulation = False
        _fill_bars(bot)
        bot._place_order = AsyncMock(return_value="order-123")

        run(bot._run_ai_prediction())

        lock_dir = os.path.join(tempfile.gettempdir(), f"algo_order_{bot.account}.lockdir")
        assert not os.path.exists(lock_dir)

    def test_lock_released_after_rejected_order(self):
        """Lock must be gone even when the broker rejects the order."""
        import os, tempfile
        bot = _make_bot(stop_pts=10.0, target_pts=20.0)
        bot._is_simulation = False
        _fill_bars(bot)
        bot._place_order = AsyncMock(return_value=None)  # broker rejection

        run(bot._run_ai_prediction())

        lock_dir = os.path.join(tempfile.gettempdir(), f"algo_order_{bot.account}.lockdir")
        assert not os.path.exists(lock_dir)

    def test_stale_lock_is_cleared_and_order_placed(self):
        """A lock older than 30s is treated as stale and removed so trading can resume."""
        import os, tempfile, time
        bot = _make_bot(stop_pts=10.0, target_pts=20.0)
        bot._is_simulation = False
        _fill_bars(bot)
        bot._place_order = AsyncMock(return_value="order-456")

        lock_dir = os.path.join(tempfile.gettempdir(), f"algo_order_{bot.account}.lockdir")
        cooldown_file = os.path.join(tempfile.gettempdir(), f"algo_order_placed_{bot.account}.txt")
        os.makedirs(lock_dir, exist_ok=True)
        # Back-date the directory's mtime by 31 seconds
        stale_mtime = time.time() - 31
        os.utime(lock_dir, (stale_mtime, stale_mtime))
        # Remove any stale cooldown file so the cooldown check doesn't block
        if os.path.exists(cooldown_file):
            os.remove(cooldown_file)
        try:
            run(bot._run_ai_prediction())
            bot._place_order.assert_called_once()
        finally:
            if os.path.exists(lock_dir):
                os.rmdir(lock_dir)
            if os.path.exists(cooldown_file):
                os.remove(cooldown_file)

    def test_fresh_lock_is_not_cleared(self):
        """A lock held for <30s is NOT treated as stale — second process is blocked."""
        import os, tempfile
        bot = _make_bot(stop_pts=10.0, target_pts=20.0)
        bot._is_simulation = False
        _fill_bars(bot)
        bot._place_order = AsyncMock(return_value="order-789")

        lock_dir = os.path.join(tempfile.gettempdir(), f"algo_order_{bot.account}.lockdir")
        os.makedirs(lock_dir, exist_ok=True)
        try:
            run(bot._run_ai_prediction())
            bot._place_order.assert_not_called()  # blocked by fresh lock
        finally:
            os.rmdir(lock_dir)


class TestSimulationBotMutexBypass:
    """SimulationBot must skip the cross-process mutex.

    Root cause of the regression: the mutex used self.account, which SimulationBot
    never set, causing AttributeError on every bar where a signal fired and making
    all backtest trades disappear (502 predict errors, 0 trades).

    Three invariants that must hold forever:
      1. SimulationBot._is_simulation is True (class-level flag)
      2. No lockdir is created or checked during a simulated signal
      3. The base class always sets self.account so live bots can't hit the same crash
    """

    def test_simulation_bot_is_simulation_flag(self):
        """SimulationBot must declare _is_simulation = True at the class level."""
        from bots.simulation_bot import SimulationBot
        assert SimulationBot._is_simulation is True

    def test_live_bot_is_simulation_flag_is_false(self):
        """TradingBot base class must have _is_simulation = False — live bots inherit this."""
        from bots.trading_bot_base import TradingBot
        assert TradingBot._is_simulation is False

    def test_simulation_bot_does_not_create_lockdir(self):
        """Running _run_ai_prediction() on a simulation bot must not touch the filesystem mutex.

        If the mutex is bypassed correctly, no lockdir is created and no AttributeError
        is raised — even if the bot has no explicit account attribute.
        """
        import os, tempfile

        class SimBot(ConcreteBot):
            _is_simulation = True

        strategy = MockStrategy()
        strategy.get_stop_target_pts = MagicMock(return_value=(10.0, 20.0))
        bot = SimBot(
            contract="MES",
            size=1,
            timeframe_minutes=5,
            strategy=strategy,
            entry_conf=0.70,
            adx_thresh=0,
            stop_pts=10.0,
            target_pts=20.0,
            risk_amount=50.0,
        )
        bot._has_existing_position = AsyncMock(return_value=False)
        bot._place_order = AsyncMock(return_value="order-sim")
        _fill_bars(bot)

        # Verify no lockdir exists before or after
        lock_dir = os.path.join(tempfile.gettempdir(), f"algo_order_{bot.account}.lockdir")
        assert not os.path.exists(lock_dir), "lockdir should not exist before run"

        run(bot._run_ai_prediction())

        assert not os.path.exists(lock_dir), "lockdir must not be created in simulation mode"
        bot._place_order.assert_called_once()  # trade still executes

    def test_simulation_bot_places_order_without_crashing(self):
        """Regression: simulation bot must not raise AttributeError on self.account.

        Before the fix, every signal in backtest mode crashed here and was silently
        counted as a 'predict_error', producing 0 trades even when signals fired.
        """
        class SimBot(ConcreteBot):
            _is_simulation = True
            # Deliberately do NOT set account — base class must provide the default

        strategy = MockStrategy()
        strategy.get_stop_target_pts = MagicMock(return_value=(10.0, 20.0))
        bot = SimBot(
            contract="MES",
            size=1,
            timeframe_minutes=5,
            strategy=strategy,
            entry_conf=0.70,
            adx_thresh=0,
            stop_pts=10.0,
            target_pts=20.0,
            risk_amount=50.0,
        )
        bot._has_existing_position = AsyncMock(return_value=False)
        bot._place_order = AsyncMock(return_value="order-sim")
        _fill_bars(bot)

        # Must not raise — before the fix this raised AttributeError: 'SimBot' has no 'account'
        run(bot._run_ai_prediction())

        assert bot.skip_stats['predict_error'] == 0, (
            "predict_error count must be 0 — AttributeError in mutex code was silently "
            "caught and counted as a predict error, hiding all backtest trades"
        )
        bot._place_order.assert_called_once()

    def test_base_class_always_sets_account(self):
        """TradingBot.__init__ must set self.account so live bots never hit AttributeError."""
        bot = _make_bot(contract="MNQ")
        assert hasattr(bot, 'account'), "base class must always set self.account"
        assert bot.account  # non-empty


class TestDoubleEntryCooldown:
    """
    Regression tests for the cross-process double-entry cooldown guard.

    Root cause: two live bot processes both saw "no position" from the broker within
    milliseconds of each other (broker state lag), and both placed orders before either's
    order was reflected. The cooldown file written after every successful order prevents
    a second process from placing within a 10-second window.
    """

    ACCOUNT = "COOLDOWN_TEST_9x7"  # unique enough to avoid collisions with other tests

    def _cooldown_path(self):
        import os, tempfile
        return os.path.join(tempfile.gettempdir(), f"algo_order_placed_{self.ACCOUNT}.txt")

    def _lock_path(self):
        import os, tempfile
        return os.path.join(tempfile.gettempdir(), f"algo_order_{self.ACCOUNT}.lockdir")

    def _make_live_bot(self):
        bot = _make_bot(stop_pts=10.0, target_pts=20.0)
        bot._is_simulation = False
        bot.account = self.ACCOUNT
        _fill_bars(bot)
        return bot

    def _cleanup(self):
        import os
        for p in [self._cooldown_path(), self._lock_path()]:
            try:
                if os.path.isfile(p):
                    os.remove(p)
                elif os.path.isdir(p):
                    os.rmdir(p)
            except Exception:
                pass

    def test_cooldown_blocks_second_order_within_window(self):
        """A cooldown file written < 10s ago must block a new order."""
        import time
        self._cleanup()
        with open(self._cooldown_path(), 'w') as f:
            f.write(str(time.time() - 2))  # 2 seconds ago
        bot = self._make_live_bot()
        try:
            run(bot._run_ai_prediction())
            bot._place_order.assert_not_called()
        finally:
            self._cleanup()

    def test_cooldown_file_written_after_successful_order(self):
        """A successful order must write the cooldown timestamp file."""
        import time, os
        self._cleanup()
        bot = self._make_live_bot()
        bot._place_order = AsyncMock(return_value="order-ok")
        try:
            run(bot._run_ai_prediction())
            assert os.path.exists(self._cooldown_path()), "cooldown file must be written"
            ts = float(open(self._cooldown_path()).read().strip())
            assert abs(time.time() - ts) < 5
        finally:
            self._cleanup()

    def test_cooldown_allows_order_after_window_expires(self):
        """A cooldown file written > 10s ago must NOT block a new order."""
        import time
        self._cleanup()
        with open(self._cooldown_path(), 'w') as f:
            f.write(str(time.time() - 15))  # 15s ago — expired
        bot = self._make_live_bot()
        bot._place_order = AsyncMock(return_value="order-ok")
        try:
            run(bot._run_ai_prediction())
            bot._place_order.assert_called_once()
        finally:
            self._cleanup()

    def test_simulation_bot_skips_cooldown_check(self):
        """ConcreteBot with _is_simulation=True must not read or write the cooldown file."""
        import os
        self._cleanup()
        bot = _make_bot(stop_pts=10.0, target_pts=20.0)  # _is_simulation=True by default
        bot.account = self.ACCOUNT
        _fill_bars(bot)
        bot._place_order = AsyncMock(return_value="order-ok")
        run(bot._run_ai_prediction())
        assert not os.path.exists(self._cooldown_path()), \
            "simulation bot must not write cooldown file"
