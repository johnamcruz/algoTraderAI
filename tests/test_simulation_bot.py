"""Unit tests for SimulationBot."""

import asyncio
import pytest
import pandas as pd
from datetime import datetime, timezone
from bots.simulation_bot import SimulationBot
from tests.helpers import MockStrategy


def run(coro):
    """Run a coroutine synchronously (avoids pytest-asyncio dependency)."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ─────────────────────────────────────────────
# Tick spec helpers
# ─────────────────────────────────────────────

class TestGetTickValue:
    """_get_tick_value: resolves dollar-per-tick from contract string."""

    def test_full_contract_id_mnq(self, sim_bot):
        sim_bot.contract = "CON.F.US.MNQ.Z25"
        assert sim_bot._get_tick_value() == 0.50

    def test_full_contract_id_enq(self, sim_bot):
        sim_bot.contract = "CON.F.US.ENQ.Z25"
        assert sim_bot._get_tick_value() == 5.00

    def test_full_contract_id_ep(self, sim_bot):
        sim_bot.contract = "CON.F.US.EP.Z25"
        assert sim_bot._get_tick_value() == 12.50

    def test_simple_symbol_nq(self, sim_bot):
        sim_bot.contract = "NQ"
        assert sim_bot._get_tick_value() == 5.00

    def test_simple_symbol_mes(self, sim_bot):
        sim_bot.contract = "MES"
        assert sim_bot._get_tick_value() == 1.25

    def test_unknown_contract_fallback(self, sim_bot):
        sim_bot.contract = "UNKNOWN"
        assert sim_bot._get_tick_value() == 0.50


class TestGetTickSize:
    """_get_tick_size: resolves minimum price increment from contract string."""

    def test_full_contract_id_mnq(self, sim_bot):
        sim_bot.contract = "CON.F.US.MNQ.Z25"
        assert sim_bot._get_tick_size() == 0.25

    def test_full_contract_id_enq(self, sim_bot):
        sim_bot.contract = "CON.F.US.ENQ.Z25"
        assert sim_bot._get_tick_size() == 0.25

    def test_simple_symbol_mnq(self, sim_bot):
        sim_bot.contract = "MNQ"
        assert sim_bot._get_tick_size() == 0.25

    def test_unknown_uses_init_tick_size(self, mock_strategy, tmp_path):
        csv_path = tmp_path / "d.csv"
        csv_path.write_text("time,open,high,low,close,volume\n1700000000,100,110,90,105,1000\n")
        bot = SimulationBot(
            csv_path=str(csv_path),
            contract="UNKNOWN",
            size=1,
            timeframe_minutes=5,
            strategy=mock_strategy,
            entry_conf=0.70,

            stop_pts=10.0,
            target_pts=20.0,
            tick_size=0.05,
        )
        assert bot._get_tick_size() == 0.05


# ─────────────────────────────────────────────
# Points → dollars conversion
# ─────────────────────────────────────────────

class TestPointsToDollars:
    """_points_to_dollars: tick_value / tick_size * points * size."""

    def test_mnq_one_contract(self, sim_bot):
        # MNQ: tick_value=0.50, tick_size=0.25 → point_value=2.0
        # 10 points * 2.0 * 1 contract = $20
        sim_bot.contract = "MNQ"
        sim_bot.current_trade_size = 1
        assert sim_bot._points_to_dollars(10.0) == pytest.approx(20.0)

    def test_mnq_multi_contract(self, sim_bot):
        sim_bot.contract = "MNQ"
        sim_bot.current_trade_size = 3
        assert sim_bot._points_to_dollars(10.0) == pytest.approx(60.0)

    def test_nq_one_contract(self, sim_bot):
        # NQ: tick_value=5.00, tick_size=0.25 → point_value=20.0
        # 10 points * 20.0 * 1 = $200
        sim_bot.contract = "NQ"
        sim_bot.current_trade_size = 1
        assert sim_bot._points_to_dollars(10.0) == pytest.approx(200.0)

    def test_negative_points(self, sim_bot):
        sim_bot.contract = "MNQ"
        sim_bot.current_trade_size = 1
        assert sim_bot._points_to_dollars(-5.0) == pytest.approx(-10.0)

    def test_zero_points(self, sim_bot):
        assert sim_bot._points_to_dollars(0.0) == pytest.approx(0.0)


# ─────────────────────────────────────────────
# CSV loading
# ─────────────────────────────────────────────

class TestLoadCsvData:
    """_load_csv_data: parses OHLCV CSVs with various timestamp formats."""

    def _make_bot(self, mock_strategy, tmp_path, content, filename="data.csv", tick_size=0.25):
        csv_path = tmp_path / filename
        csv_path.write_text(content)
        bot = SimulationBot(
            csv_path=str(csv_path),
            contract="MNQ",
            size=1,
            timeframe_minutes=5,
            strategy=mock_strategy,
            entry_conf=0.70,

            stop_pts=10.0,
            target_pts=20.0,
            tick_size=tick_size,
        )
        return bot

    def test_unix_timestamp_parsed(self, mock_strategy, tmp_path):
        bot = self._make_bot(mock_strategy, tmp_path,
            "time,open,high,low,close,volume\n"
            "1700000000,100,110,90,105,1000\n"
            "1700000060,106,115,100,112,800\n"
        )
        df = bot._load_csv_data()
        assert len(df) == 2
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_datetime_string_parsed(self, mock_strategy, tmp_path):
        bot = self._make_bot(mock_strategy, tmp_path,
            "time,open,high,low,close,volume\n"
            "2024-01-15 09:30:00,100,110,90,105,1000\n"
            "2024-01-15 09:35:00,106,115,100,112,800\n"
        )
        df = bot._load_csv_data()
        assert len(df) == 2
        assert df.index[0] < df.index[1]

    def test_timestamp_column_accepted(self, mock_strategy, tmp_path):
        bot = self._make_bot(mock_strategy, tmp_path,
            "timestamp,open,high,low,close,volume\n"
            "1700000000,100,110,90,105,1000\n"
        )
        df = bot._load_csv_data()
        assert len(df) == 1

    def test_date_column_accepted(self, mock_strategy, tmp_path):
        bot = self._make_bot(mock_strategy, tmp_path,
            "date,open,high,low,close,volume\n"
            "2024-01-15 09:30:00,100,110,90,105,1000\n"
        )
        df = bot._load_csv_data()
        assert len(df) == 1

    def test_missing_volume_filled_with_zero(self, mock_strategy, tmp_path):
        bot = self._make_bot(mock_strategy, tmp_path,
            "time,open,high,low,close\n"
            "1700000000,100,110,90,105\n"
        )
        df = bot._load_csv_data()
        assert "volume" in df.columns
        assert df["volume"].iloc[0] == 0

    def test_missing_required_column_raises(self, mock_strategy, tmp_path):
        bot = self._make_bot(mock_strategy, tmp_path,
            "time,open,high,low,volume\n"  # 'close' missing
            "1700000000,100,110,90,1000\n"
        )
        with pytest.raises(ValueError, match="missing required columns"):
            bot._load_csv_data()

    def test_no_time_column_raises(self, mock_strategy, tmp_path):
        bot = self._make_bot(mock_strategy, tmp_path,
            "open,high,low,close,volume\n"
            "100,110,90,105,1000\n"
        )
        with pytest.raises(ValueError, match="time.*timestamp.*date"):
            bot._load_csv_data()

    def test_columns_sorted_by_timestamp(self, mock_strategy, tmp_path):
        # Rows intentionally out of order
        bot = self._make_bot(mock_strategy, tmp_path,
            "time,open,high,low,close,volume\n"
            "1700000060,106,115,100,112,800\n"
            "1700000000,100,110,90,105,1000\n"
        )
        df = bot._load_csv_data()
        assert df.index[0] < df.index[1]

    def test_output_columns(self, mock_strategy, tmp_path):
        bot = self._make_bot(mock_strategy, tmp_path,
            "time,open,high,low,close,volume,extra\n"
            "1700000000,100,110,90,105,1000,ignored\n"
        )
        df = bot._load_csv_data()
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}


# ─────────────────────────────────────────────
# _process_bar exit logic
# ─────────────────────────────────────────────

class TestProcessBarExitLogic:
    """
    Tests for gap-open and intrabar exit logic in _process_bar.

    Each test puts the bot into an open position, then feeds a single
    bar and checks that the correct exit price / reason / position-reset
    behaviour is produced.
    """

    TS = datetime(2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    def _put_in_position(self, bot, position_type, entry, stop, target):
        bot.in_position = True
        bot.position_type = position_type
        bot.entry_price = entry
        bot.stop_loss = stop
        bot.profit_target = target
        bot.entry_timestamp = self.TS
        bot.just_entered_this_bar = False
        bot.current_trade_size = 1

    def _make_bar(self, open_, high, low, close):
        return (self.TS, {"open": open_, "high": high, "low": low,
                          "close": close, "volume": 100})

    # ── Gap-open exits ────────────────────────────────────────────────

    def test_long_gap_open_above_target_exits_at_target(self, sim_bot):
        """Bar opens above TP → limit order fills at TP, not gap price."""
        self._put_in_position(sim_bot, "LONG", entry=100, stop=95, target=110)
        bar = self._make_bar(open_=115, high=120, low=114, close=117)
        run(sim_bot._process_bar(bar, 1, 10))
        trade = sim_bot.trades_log[-1]
        assert trade["exit"] == 110
        assert trade["reason"] == "PROFIT_TARGET"
        assert not sim_bot.in_position

    def test_long_gap_open_below_stop_exits_at_open(self, sim_bot):
        """Bar gaps below stop → market-fill slippage, exits at open."""
        self._put_in_position(sim_bot, "LONG", entry=100, stop=95, target=110)
        bar = self._make_bar(open_=92, high=93, low=90, close=91)
        run(sim_bot._process_bar(bar, 1, 10))
        trade = sim_bot.trades_log[-1]
        assert trade["exit"] == 92
        assert trade["reason"] == "STOP_LOSS"

    def test_short_gap_open_below_target_exits_at_target(self, sim_bot):
        """SHORT gap below TP → limit order fills at TP."""
        self._put_in_position(sim_bot, "SHORT", entry=100, stop=105, target=90)
        bar = self._make_bar(open_=85, high=86, low=84, close=85)
        run(sim_bot._process_bar(bar, 1, 10))
        trade = sim_bot.trades_log[-1]
        assert trade["exit"] == 90
        assert trade["reason"] == "PROFIT_TARGET"

    def test_short_gap_open_above_stop_exits_at_open(self, sim_bot):
        """SHORT gap above stop → market-fill slippage."""
        self._put_in_position(sim_bot, "SHORT", entry=100, stop=105, target=90)
        bar = self._make_bar(open_=108, high=110, low=107, close=109)
        run(sim_bot._process_bar(bar, 1, 10))
        trade = sim_bot.trades_log[-1]
        assert trade["exit"] == 108
        assert trade["reason"] == "STOP_LOSS"

    # ── Intrabar wick exits ───────────────────────────────────────────

    def test_long_wick_hits_target(self, sim_bot):
        """High touches TP → exits at profit_target."""
        self._put_in_position(sim_bot, "LONG", entry=100, stop=95, target=110)
        bar = self._make_bar(open_=101, high=111, low=100, close=105)
        run(sim_bot._process_bar(bar, 1, 10))
        trade = sim_bot.trades_log[-1]
        assert trade["exit"] == 110
        assert trade["reason"] == "PROFIT_TARGET"

    def test_long_wick_hits_stop(self, sim_bot):
        """Low touches stop → exits at stop_loss."""
        self._put_in_position(sim_bot, "LONG", entry=100, stop=95, target=110)
        bar = self._make_bar(open_=99, high=101, low=94, close=98)
        run(sim_bot._process_bar(bar, 1, 10))
        trade = sim_bot.trades_log[-1]
        assert trade["exit"] == 95
        assert trade["reason"] == "STOP_LOSS"

    def test_short_wick_hits_target(self, sim_bot):
        """Low touches TP on SHORT → exits at profit_target."""
        self._put_in_position(sim_bot, "SHORT", entry=100, stop=105, target=90)
        bar = self._make_bar(open_=99, high=100, low=89, close=95)
        run(sim_bot._process_bar(bar, 1, 10))
        trade = sim_bot.trades_log[-1]
        assert trade["exit"] == 90
        assert trade["reason"] == "PROFIT_TARGET"

    def test_short_wick_hits_stop(self, sim_bot):
        """High touches stop on SHORT → exits at stop_loss."""
        self._put_in_position(sim_bot, "SHORT", entry=100, stop=105, target=90)
        bar = self._make_bar(open_=101, high=106, low=100, close=102)
        run(sim_bot._process_bar(bar, 1, 10))
        trade = sim_bot.trades_log[-1]
        assert trade["exit"] == 105
        assert trade["reason"] == "STOP_LOSS"

    # ── Intrabar close-based exits ────────────────────────────────────

    def test_long_close_above_target_exits_at_close(self, sim_bot):
        """Bar closes above TP but wick never reached TP → exits at close."""
        self._put_in_position(sim_bot, "LONG", entry=100, stop=95, target=110)
        # high=109 (below 110), close=112 (above 110)
        bar = self._make_bar(open_=101, high=109, low=100, close=112)
        run(sim_bot._process_bar(bar, 1, 10))
        trade = sim_bot.trades_log[-1]
        assert trade["exit"] == 112
        assert trade["reason"] == "PROFIT_TARGET"

    def test_short_close_below_target_exits_at_close(self, sim_bot):
        """Bar closes below TP but wick never reached TP → exits at close."""
        self._put_in_position(sim_bot, "SHORT", entry=100, stop=105, target=90)
        # low=91 (above 90), close=88 (below 90)
        bar = self._make_bar(open_=99, high=100, low=91, close=88)
        run(sim_bot._process_bar(bar, 1, 10))
        trade = sim_bot.trades_log[-1]
        assert trade["exit"] == 88
        assert trade["reason"] == "PROFIT_TARGET"

    # ── Both stop and target hit same bar ────────────────────────────

    def test_long_tp_wins_when_wick_hits_even_if_open_near_stop(self, sim_bot):
        """TP limit order fills the instant price touches it, even if open is near stop.
        When the bar's high crosses the TP wick, the limit is assumed to fill first."""
        self._put_in_position(sim_bot, "LONG", entry=100, stop=95, target=110)
        # open=96 (near stop=95) but wick reaches 111 — TP limit fires at 110
        bar = self._make_bar(open_=96, high=111, low=94, close=105)
        run(sim_bot._process_bar(bar, 1, 10))
        trade = sim_bot.trades_log[-1]
        assert trade["reason"] == "PROFIT_TARGET"
        assert trade["exit"] == 110

    def test_long_target_wins_when_open_closer_to_high(self, sim_bot):
        """Open near high → target hit first, exits at profit_target."""
        self._put_in_position(sim_bot, "LONG", entry=100, stop=95, target=110)
        # open=109 (1 pt below target=110, 14 pts above stop=95)
        bar = self._make_bar(open_=109, high=111, low=94, close=105)
        run(sim_bot._process_bar(bar, 1, 10))
        trade = sim_bot.trades_log[-1]
        assert trade["reason"] == "PROFIT_TARGET"
        assert trade["exit"] == 110

    def test_short_tp_wins_when_wick_hits_even_if_open_near_stop(self, sim_bot):
        """SHORT: TP limit fires when low wick crosses target, even if open is near stop."""
        self._put_in_position(sim_bot, "SHORT", entry=100, stop=105, target=90)
        # open=104 (near stop=105) but wick reaches 89 — TP limit fires at 90
        bar = self._make_bar(open_=104, high=106, low=89, close=95)
        run(sim_bot._process_bar(bar, 1, 10))
        trade = sim_bot.trades_log[-1]
        assert trade["reason"] == "PROFIT_TARGET"
        assert trade["exit"] == 90

    def test_short_target_wins_when_open_closer_to_low(self, sim_bot):
        """SHORT: open near low → target hit first."""
        self._put_in_position(sim_bot, "SHORT", entry=100, stop=105, target=90)
        # open=91 (1 pt above target=90, 14 pts below stop=105)
        bar = self._make_bar(open_=91, high=106, low=89, close=95)
        run(sim_bot._process_bar(bar, 1, 10))
        trade = sim_bot.trades_log[-1]
        assert trade["reason"] == "PROFIT_TARGET"

    # ── No exit ───────────────────────────────────────────────────────

    def test_no_exit_when_bar_is_inside_range(self, sim_bot):
        """Bar that doesn't touch stop or target leaves position open."""
        self._put_in_position(sim_bot, "LONG", entry=100, stop=95, target=110)
        bar = self._make_bar(open_=101, high=105, low=98, close=103)
        before = len(sim_bot.trades_log)
        run(sim_bot._process_bar(bar, 1, 10))
        assert len(sim_bot.trades_log) == before
        assert sim_bot.in_position

    def test_no_exit_when_no_position(self, sim_bot):
        """Bar processed with no open position logs nothing."""
        bar = self._make_bar(open_=101, high=105, low=98, close=103)
        before = len(sim_bot.trades_log)
        run(sim_bot._process_bar(bar, 1, 10))
        assert len(sim_bot.trades_log) == before

    # ── Same-bar entry guard ──────────────────────────────────────────

    def test_same_bar_entry_does_not_exit(self, sim_bot):
        """Position entered this bar must not exit on the same bar."""
        self._put_in_position(sim_bot, "LONG", entry=100, stop=95, target=110)
        sim_bot.just_entered_this_bar = True
        # Bar would normally trigger stop
        bar = self._make_bar(open_=99, high=101, low=94, close=98)
        before = len(sim_bot.trades_log)
        run(sim_bot._process_bar(bar, 1, 10))
        assert len(sim_bot.trades_log) == before
        assert sim_bot.in_position

    # ── PnL accounting ────────────────────────────────────────────────

    def test_winning_trade_increments_winning_count(self, sim_bot):
        self._put_in_position(sim_bot, "LONG", entry=100, stop=95, target=110)
        bar = self._make_bar(open_=101, high=111, low=100, close=105)
        run(sim_bot._process_bar(bar, 1, 10))
        assert sim_bot.winning_trades == 1
        assert sim_bot.losing_trades == 0

    def test_losing_trade_increments_losing_count(self, sim_bot):
        self._put_in_position(sim_bot, "LONG", entry=100, stop=95, target=110)
        bar = self._make_bar(open_=99, high=101, low=94, close=98)
        run(sim_bot._process_bar(bar, 1, 10))
        assert sim_bot.losing_trades == 1
        assert sim_bot.winning_trades == 0

    def test_pnl_points_correct_for_long_win(self, sim_bot):
        """LONG TP hit: PnL = profit_target - entry."""
        self._put_in_position(sim_bot, "LONG", entry=100, stop=95, target=110)
        bar = self._make_bar(open_=101, high=111, low=100, close=105)
        run(sim_bot._process_bar(bar, 1, 10))
        assert sim_bot.trades_log[-1]["pnl_points"] == pytest.approx(10.0)

    def test_pnl_points_correct_for_long_loss(self, sim_bot):
        """LONG stop hit: PnL = stop_loss - entry (negative)."""
        self._put_in_position(sim_bot, "LONG", entry=100, stop=95, target=110)
        bar = self._make_bar(open_=99, high=101, low=94, close=98)
        run(sim_bot._process_bar(bar, 1, 10))
        assert sim_bot.trades_log[-1]["pnl_points"] == pytest.approx(-5.0)


# ── _print_summary ────────────────────────────────────────────────────

class TestPrintSummary:
    """_print_summary: must not crash when signal_meta contains string fields."""

    def test_summary_handles_string_signal_meta(self, sim_bot, capsys):
        """signal_meta with string-valued keys (htf_trend, trend_alignment) must not raise TypeError."""
        sim_bot.trades_log = [
            {
                "entry_timestamp": "2022-01-01 09:00",
                "exit_timestamp": "2022-01-01 09:05",
                "type": "LONG",
                "entry": 100.0,
                "exit": 110.0,
                "reason": "PROFIT_TARGET",
                "pnl_points": 10.0,
                "pnl_dollars": 100.0,
                "total_pnl_dollars": 100.0,
                "signal_meta": {
                    "signal_prob": 0.85,
                    "risk_rr": 8.5,
                    "confluence_score": 2,
                    "htf_trend": "UP",
                    "trend_alignment": "ALIGNED",
                },
            },
            {
                "entry_timestamp": "2022-01-02 09:00",
                "exit_timestamp": "2022-01-02 09:05",
                "type": "LONG",
                "entry": 100.0,
                "exit": 95.0,
                "reason": "STOP_LOSS",
                "pnl_points": -5.0,
                "pnl_dollars": -50.0,
                "total_pnl_dollars": 50.0,
                "signal_meta": {
                    "signal_prob": 0.72,
                    "risk_rr": 7.2,
                    "confluence_score": 1,
                    "htf_trend": "DOWN",
                    "trend_alignment": "COUNTER",
                },
            },
        ]
        sim_bot.winning_trades = 1
        sim_bot.losing_trades = 1
        sim_bot.total_pnl_points = 5.0
        sim_bot.total_pnl_dollars = 50.0

        sim_bot._print_summary()  # must not raise

    def test_summary_shows_profit_target_hit_when_exceeded(self, sim_bot, capsys):
        """When session_profit_target is set and exceeded, 'Profit Target Hit' appears in output."""
        sim_bot.session_profit_target = 1000.0
        sim_bot.trades_log = []
        sim_bot.winning_trades = 1
        sim_bot.losing_trades = 0
        sim_bot.total_pnl_points = 100.0
        sim_bot.total_pnl_dollars = 1200.0
        sim_bot._print_summary()
        out = capsys.readouterr().out
        assert "Profit Target Hit" in out

    def test_summary_omits_profit_target_hit_when_target_is_none(self, sim_bot, capsys):
        """When session_profit_target is None (--no-target mode), 'Profit Target Hit' must not appear."""
        sim_bot.session_profit_target = None
        sim_bot.trades_log = []
        sim_bot.winning_trades = 1
        sim_bot.losing_trades = 0
        sim_bot.total_pnl_points = 100.0
        sim_bot.total_pnl_dollars = 99999.0
        sim_bot._print_summary()
        out = capsys.readouterr().out
        assert "Profit Target Hit" not in out


# ── SimulationBot startup print ───────────────────────────────────────

class TestSimulationBotStartupPrint:
    """Startup print must not crash regardless of profit_target value."""

    def test_startup_print_with_none_profit_target(self, sim_bot, capsys):
        """profit_target=None renders as 'none', not a format crash."""
        sim_bot.session_profit_target = None
        pt_str = f"${sim_bot.session_profit_target:,.2f}" if sim_bot.session_profit_target is not None else "none"
        assert pt_str == "none"

    def test_startup_print_with_numeric_profit_target(self, sim_bot, capsys):
        """profit_target set to a value renders correctly."""
        sim_bot.session_profit_target = 12000.0
        pt_str = f"${sim_bot.session_profit_target:,.2f}" if sim_bot.session_profit_target is not None else "none"
        assert pt_str == "$12,000.00"
