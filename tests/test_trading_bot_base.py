"""Unit tests for TradingBot base class logic."""

import pytest
from tests.helpers import ConcreteBot, MockStrategy


class TestCalculateSize:
    """_calculate_size: dynamic sizing based on risk budget."""

    def test_risk_fits_exactly_one_contract(self, bot):
        # 50 risk / (100 ticks * $0.50) = 1.0 → 1 contract
        assert bot._calculate_size(sl_ticks=100) == 1

    def test_risk_capped_at_15(self, bot):
        # 50 / (1 * 0.50) = 100 → capped at 15
        assert bot._calculate_size(sl_ticks=1) == 15

    def test_risk_floors_to_zero_when_stop_too_wide(self, bot):
        # 50 / (1000 * 0.50) = 0.1 → floor → 0
        assert bot._calculate_size(sl_ticks=1000) == 0

    def test_zero_ticks_returns_zero(self, bot):
        assert bot._calculate_size(sl_ticks=0) == 0

    def test_negative_ticks_treated_as_absolute(self, bot):
        # LONG uses negative stop_ticks; abs() normalises them
        # 50 / (100 * 0.50) = 1
        assert bot._calculate_size(sl_ticks=-100) == 1

    def test_effective_risk_override(self, bot):
        # Pass a custom effective_risk instead of self.risk_amount
        # 200 / (100 * 0.50) = 4
        assert bot._calculate_size(sl_ticks=100, effective_risk=200.0) == 4

    def test_falls_back_to_size_when_no_risk_amount(self, mock_strategy):
        b = ConcreteBot(
            contract="MNQ",
            size=3,
            timeframe_minutes=5,
            strategy=mock_strategy,
            entry_conf=0.70,

            stop_pts=10.0,
            target_pts=20.0,
            risk_amount=None,
        )
        assert b._calculate_size(sl_ticks=10) == 3

    def test_high_conf_multiplier_scales_size(self, mock_strategy):
        # With risk=50, multiplier=2 → effective_risk=100
        # 100 / (100 * 0.50) = 2
        b = ConcreteBot(
            contract="MNQ",
            size=1,
            timeframe_minutes=5,
            strategy=mock_strategy,
            entry_conf=0.70,

            stop_pts=10.0,
            target_pts=20.0,
            risk_amount=50.0,
        )
        assert b._calculate_size(sl_ticks=100, effective_risk=100.0) == 2


class TestCalculatePnl:
    """_calculate_pnl: point P&L for long and short positions."""

    def test_long_profit(self, bot):
        bot.position_type = "LONG"
        bot.entry_price = 100.0
        assert bot._calculate_pnl(exit_price=110.0) == pytest.approx(10.0)

    def test_long_loss(self, bot):
        bot.position_type = "LONG"
        bot.entry_price = 100.0
        assert bot._calculate_pnl(exit_price=90.0) == pytest.approx(-10.0)

    def test_short_profit(self, bot):
        bot.position_type = "SHORT"
        bot.entry_price = 100.0
        assert bot._calculate_pnl(exit_price=85.0) == pytest.approx(15.0)

    def test_short_loss(self, bot):
        bot.position_type = "SHORT"
        bot.entry_price = 100.0
        assert bot._calculate_pnl(exit_price=110.0) == pytest.approx(-10.0)

    def test_breakeven(self, bot):
        bot.position_type = "LONG"
        bot.entry_price = 100.0
        assert bot._calculate_pnl(exit_price=100.0) == pytest.approx(0.0)

    def test_no_position_returns_zero(self, bot):
        bot.position_type = None
        bot.entry_price = 100.0
        assert bot._calculate_pnl(exit_price=110.0) == pytest.approx(0.0)


class TestCheckExitConditions:
    """_check_exit_conditions: stop loss and profit target detection."""

    def _setup_long(self, bot):
        bot.in_position = True
        bot.position_type = "LONG"
        bot.entry_price = 100.0
        bot.stop_loss = 95.0
        bot.profit_target = 110.0

    def _setup_short(self, bot):
        bot.in_position = True
        bot.position_type = "SHORT"
        bot.entry_price = 100.0
        bot.stop_loss = 105.0
        bot.profit_target = 90.0

    def test_not_in_position_returns_none(self, bot):
        bot.in_position = False
        assert bot._check_exit_conditions(99.0) == (None, None)

    def test_missing_stop_loss_returns_none(self, bot):
        bot.in_position = True
        bot.stop_loss = None
        bot.profit_target = 110.0
        assert bot._check_exit_conditions(99.0) == (None, None)

    def test_long_stop_loss_hit(self, bot):
        self._setup_long(bot)
        price, reason = bot._check_exit_conditions(94.0)
        assert price == 95.0
        assert reason == "STOP_LOSS"

    def test_long_stop_loss_at_exact_level(self, bot):
        self._setup_long(bot)
        price, reason = bot._check_exit_conditions(95.0)
        assert reason == "STOP_LOSS"

    def test_long_profit_target_hit(self, bot):
        self._setup_long(bot)
        price, reason = bot._check_exit_conditions(111.0)
        assert price == 110.0
        assert reason == "PROFIT_TARGET"

    def test_long_profit_target_at_exact_level(self, bot):
        self._setup_long(bot)
        price, reason = bot._check_exit_conditions(110.0)
        assert reason == "PROFIT_TARGET"

    def test_long_no_exit_between_levels(self, bot):
        self._setup_long(bot)
        assert bot._check_exit_conditions(102.0) == (None, None)

    def test_short_stop_loss_hit(self, bot):
        self._setup_short(bot)
        price, reason = bot._check_exit_conditions(106.0)
        assert price == 105.0
        assert reason == "STOP_LOSS"

    def test_short_profit_target_hit(self, bot):
        self._setup_short(bot)
        price, reason = bot._check_exit_conditions(89.0)
        assert price == 90.0
        assert reason == "PROFIT_TARGET"

    def test_short_no_exit_between_levels(self, bot):
        self._setup_short(bot)
        assert bot._check_exit_conditions(98.0) == (None, None)


class TestBreakeven:
    """_check_and_set_breakeven: moves stop to entry once 1R profit is reached."""

    def _long(self, bot, entry=100.0, stop=95.0, target=110.0):
        bot.in_position = True
        bot.position_type = "LONG"
        bot.entry_price = entry
        bot.stop_loss = stop
        bot.profit_target = target
        bot.breakeven_set = False
        bot.breakeven_on_2r = True

    def _short(self, bot, entry=100.0, stop=105.0, target=90.0):
        bot.in_position = True
        bot.position_type = "SHORT"
        bot.entry_price = entry
        bot.stop_loss = stop
        bot.profit_target = target
        bot.breakeven_set = False
        bot.breakeven_on_2r = True

    # ── Disabled ──────────────────────────────────────────────────────────────

    def test_disabled_when_flag_off(self, bot):
        self._long(bot)
        bot.breakeven_on_2r = False
        triggered = bot._check_and_set_breakeven(110.0)
        assert triggered is False
        assert bot.stop_loss == pytest.approx(95.0)  # unchanged

    def test_disabled_when_predicted_rr_below_3(self, bot):
        # 2R trades (predicted_rr=2) should never trigger breakeven — TP is at 2R
        self._long(bot)
        bot.strategy._latest_risk_rr = 2.0
        triggered = bot._check_and_set_breakeven(110.0)
        assert triggered is False
        assert bot.stop_loss == pytest.approx(95.0)
        bot.strategy._latest_risk_rr = 3.0  # restore default

    def test_disabled_when_predicted_rr_exactly_3(self, bot):
        # predicted_rr=3.0 is the minimum that qualifies; should trigger at 2R
        self._long(bot)
        bot.strategy._latest_risk_rr = 3.0
        triggered = bot._check_and_set_breakeven(110.0)
        assert triggered is True

    # ── LONG ──────────────────────────────────────────────────────────────────

    def test_long_not_triggered_below_2r(self, bot):
        # entry=100, stop=95 → stop_dist=5 → 2R at 110
        self._long(bot)
        triggered = bot._check_and_set_breakeven(109.99)
        assert triggered is False
        assert bot.stop_loss == pytest.approx(95.0)

    def test_long_not_triggered_at_1r(self, bot):
        # 1R (105) must no longer trigger breakeven
        self._long(bot)
        triggered = bot._check_and_set_breakeven(105.0)
        assert triggered is False
        assert bot.stop_loss == pytest.approx(95.0)

    def test_long_triggered_exactly_at_2r(self, bot):
        self._long(bot)
        triggered = bot._check_and_set_breakeven(110.0)
        assert triggered is True
        assert bot.stop_loss == pytest.approx(100.0)  # moved to entry
        assert bot.breakeven_set is True

    def test_long_triggered_above_2r(self, bot):
        self._long(bot)
        triggered = bot._check_and_set_breakeven(113.0)
        assert triggered is True
        assert bot.stop_loss == pytest.approx(100.0)

    def test_long_only_triggers_once(self, bot):
        self._long(bot)
        bot._check_and_set_breakeven(111.0)   # triggers at 2R
        bot.stop_loss = 100.0                  # already moved
        triggered = bot._check_and_set_breakeven(115.0)  # should not fire again
        assert triggered is False

    # ── SHORT ─────────────────────────────────────────────────────────────────

    def test_short_not_triggered_above_2r(self, bot):
        # entry=100, stop=105 → stop_dist=5 → 2R at 90
        self._short(bot)
        triggered = bot._check_and_set_breakeven(90.01)
        assert triggered is False
        assert bot.stop_loss == pytest.approx(105.0)

    def test_short_not_triggered_at_1r(self, bot):
        # 1R (95) must no longer trigger breakeven
        self._short(bot)
        triggered = bot._check_and_set_breakeven(95.0)
        assert triggered is False
        assert bot.stop_loss == pytest.approx(105.0)

    def test_short_triggered_exactly_at_2r(self, bot):
        self._short(bot)
        triggered = bot._check_and_set_breakeven(90.0)
        assert triggered is True
        assert bot.stop_loss == pytest.approx(100.0)
        assert bot.breakeven_set is True

    def test_short_triggered_below_2r(self, bot):
        self._short(bot)
        triggered = bot._check_and_set_breakeven(87.0)
        assert triggered is True
        assert bot.stop_loss == pytest.approx(100.0)

    # ── Edge cases ─────────────────────────────────────────────────────────────

    def test_no_position_is_noop(self, bot):
        bot.in_position = False
        bot.breakeven_on_2r = True
        assert bot._check_and_set_breakeven(110.0) is False

    def test_breakeven_resets_with_position_state(self, bot):
        self._long(bot)
        bot._check_and_set_breakeven(111.0)   # 2R trigger
        assert bot.breakeven_set is True
        bot._reset_position_state()
        assert bot.breakeven_set is False

    def test_short_only_triggers_once(self, bot):
        self._short(bot)
        bot._check_and_set_breakeven(92.0)    # triggers
        bot.stop_loss = 100.0
        triggered = bot._check_and_set_breakeven(88.0)   # must not fire again
        assert triggered is False

    def test_stop_dist_zero_is_noop(self, bot):
        """stop == entry → division-by-zero guard → no trigger."""
        self._long(bot, entry=100.0, stop=100.0, target=110.0)
        triggered = bot._check_and_set_breakeven(110.0)
        assert triggered is False
        assert bot.stop_loss == pytest.approx(100.0)

    def test_entry_price_none_is_noop(self, bot):
        self._long(bot)
        bot.entry_price = None
        assert bot._check_and_set_breakeven(110.0) is False

    def test_stop_loss_none_is_noop(self, bot):
        self._long(bot)
        bot.stop_loss = None
        assert bot._check_and_set_breakeven(110.0) is False

    def test_unknown_position_type_is_noop(self, bot):
        self._long(bot)
        bot.position_type = "FLAT"
        assert bot._check_and_set_breakeven(110.0) is False

    def test_long_stop_moved_to_entry_not_below(self, bot):
        """Stop must land exactly at entry, not above or below."""
        # entry=7168.75, stop=7163.27 → stop_dist=5.48 → 2R at 7179.71
        self._long(bot, entry=7168.75, stop=7163.27, target=7190.0)
        bot._check_and_set_breakeven(7180.0)   # past 2R
        assert bot.stop_loss == pytest.approx(7168.75)

    def test_short_stop_moved_to_entry_not_above(self, bot):
        # entry=4750, stop=4760 → stop_dist=10 → 2R at 4730
        self._short(bot, entry=4750.0, stop=4760.0, target=4720.0)
        bot._check_and_set_breakeven(4729.0)   # past 2R
        assert bot.stop_loss == pytest.approx(4750.0)

    def test_2r_threshold_is_stop_distance_not_target(self, bot):
        """2R = 2 × stop distance, not the profit target."""
        # entry=100, stop=95 (5pt), target=120 (20pt) → 2R triggers at 110
        self._long(bot, entry=100.0, stop=95.0, target=120.0)
        assert bot._check_and_set_breakeven(109.99) is False
        assert bot._check_and_set_breakeven(110.0)  is True


class TestUpdateMfe:
    """_update_mfe: tracks peak unrealized gain in points."""

    def _long(self, bot):
        bot.in_position = True
        bot.position_type = "LONG"
        bot.entry_price = 100.0
        bot.mfe_pts = 0.0

    def _short(self, bot):
        bot.in_position = True
        bot.position_type = "SHORT"
        bot.entry_price = 100.0
        bot.mfe_pts = 0.0

    def test_long_favorable_tick_recorded(self, bot):
        self._long(bot)
        bot._update_mfe(103.0)
        assert bot.mfe_pts == pytest.approx(3.0)

    def test_long_peak_not_overwritten_by_lower_tick(self, bot):
        self._long(bot)
        bot._update_mfe(105.0)
        bot._update_mfe(102.0)
        assert bot.mfe_pts == pytest.approx(5.0)

    def test_long_adverse_tick_ignored(self, bot):
        self._long(bot)
        bot._update_mfe(98.0)  # below entry — no favorable movement
        assert bot.mfe_pts == pytest.approx(0.0)

    def test_short_favorable_tick_recorded(self, bot):
        self._short(bot)
        bot._update_mfe(96.0)
        assert bot.mfe_pts == pytest.approx(4.0)

    def test_short_peak_not_overwritten_by_higher_tick(self, bot):
        self._short(bot)
        bot._update_mfe(94.0)
        bot._update_mfe(97.0)
        assert bot.mfe_pts == pytest.approx(6.0)

    def test_short_adverse_tick_ignored(self, bot):
        self._short(bot)
        bot._update_mfe(102.0)
        assert bot.mfe_pts == pytest.approx(0.0)

    def test_no_position_is_noop(self, bot):
        bot.in_position = False
        bot.mfe_pts = 0.0
        bot._update_mfe(110.0)
        assert bot.mfe_pts == pytest.approx(0.0)

    def test_mfe_resets_on_position_reset(self, bot):
        self._long(bot)
        bot._update_mfe(108.0)
        assert bot.mfe_pts == pytest.approx(8.0)
        bot._reset_position_state()
        assert bot.mfe_pts == pytest.approx(0.0)


class TestResetPositionState:
    """_reset_position_state: clears all position fields."""

    def test_all_fields_cleared(self, bot):
        bot.in_position = True
        bot.position_type = "LONG"
        bot.entry_price = 100.0
        bot.stop_loss = 95.0
        bot.profit_target = 110.0
        bot.stop_orderId = "abc"
        bot.limit_orderId = "def"
        bot.stop_bracket_order_id = 99999
        bot.position_size = 3
        bot.mfe_pts = 7.5
        bot.breakeven_set = True

        bot._reset_position_state()

        assert bot.in_position is False
        assert bot.position_type is None
        assert bot.entry_price is None
        assert bot.stop_loss is None
        assert bot.profit_target is None
        assert bot.stop_orderId is None
        assert bot.limit_orderId is None
        assert bot.stop_bracket_order_id is None
        assert bot.position_size is None
        assert bot.mfe_pts == pytest.approx(0.0)
        assert bot.breakeven_set is False

    def test_new_trade_starts_with_clean_bracket_state(self, bot):
        bot.stop_bracket_order_id = 12345
        bot.position_size = 2
        bot._reset_position_state()
        assert bot.stop_bracket_order_id is None
        assert bot.position_size is None


class TestWarmupLength:
    """num_historical_candles_needed is driven by the strategy, not hardcoded."""

    def _make_bot(self, strategy):
        return ConcreteBot(
            contract="MNQ", size=1, timeframe_minutes=5,
            strategy=strategy, entry_conf=0.70,
            stop_pts=10.0, target_pts=20.0,
        )

    def test_warmup_count_matches_strategy(self, mock_strategy):
        bot = self._make_bot(mock_strategy)
        assert bot.num_historical_candles_needed == mock_strategy.get_warmup_length()

    def test_historical_bars_deque_capacity_matches_warmup(self, mock_strategy):
        bot = self._make_bot(mock_strategy)
        assert bot.historical_bars.maxlen == mock_strategy.get_warmup_length()

    def test_custom_warmup_length_propagates(self):
        strategy = MockStrategy()
        strategy.get_warmup_length = lambda: 50
        bot = self._make_bot(strategy)
        assert bot.num_historical_candles_needed == 50
        assert bot.historical_bars.maxlen == 50
