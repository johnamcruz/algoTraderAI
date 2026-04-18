"""Unit tests for TradingBot base class logic."""

import pytest
from tests.helpers import ConcreteBot


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
            adx_thresh=20.0,
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
            adx_thresh=20.0,
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

        bot._reset_position_state()

        assert bot.in_position is False
        assert bot.position_type is None
        assert bot.entry_price is None
        assert bot.stop_loss is None
        assert bot.profit_target is None
        assert bot.stop_orderId is None
        assert bot.limit_orderId is None
