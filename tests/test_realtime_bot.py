"""Unit tests for RealTimeBot (trading_bot.py)."""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from tests.helpers import MockStrategy


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def live_bot(mock_strategy):
    """RealTimeBot with SignalRClient patched out."""
    with patch("trading_bot.SignalRClient"):
        from trading_bot import RealTimeBot
        bot = RealTimeBot(
            token="test-token",
            market_hub="ws://localhost/hub",
            base_url="http://localhost",
            account="ACC1",
            contract="CON.F.US.MNQ.Z25",
            size=1,
            timeframe_minutes=5,
            strategy=mock_strategy,
            entry_conf=0.70,
            adx_thresh=20.0,
            stop_pts=10.0,
            target_pts=20.0,
            risk_amount=50.0,
        )
        # Pre-populate a contracts list so tick/size lookups work
        bot.contracts = [
            {
                "id": "CON.F.US.MNQ.Z25",
                "name": "Micro E-mini Nasdaq MNQ Dec 2025",
                "tickSize": 0.25,
                "tickValue": 0.50,
            }
        ]
        return bot


# ── _get_bar_time ──────────────────────────────────────────────────────────────

class TestGetBarTime:
    """Rounds timestamp down to nearest timeframe boundary."""

    def test_already_on_boundary(self, live_bot):
        ts = datetime(2025, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        assert live_bot._get_bar_time(ts) == datetime(2025, 1, 1, 9, 30, 0, tzinfo=timezone.utc)

    def test_rounds_down_mid_bar(self, live_bot):
        ts = datetime(2025, 1, 1, 9, 33, 45, tzinfo=timezone.utc)
        assert live_bot._get_bar_time(ts) == datetime(2025, 1, 1, 9, 30, 0, tzinfo=timezone.utc)

    def test_rounds_down_last_second_of_bar(self, live_bot):
        ts = datetime(2025, 1, 1, 9, 34, 59, tzinfo=timezone.utc)
        assert live_bot._get_bar_time(ts) == datetime(2025, 1, 1, 9, 30, 0, tzinfo=timezone.utc)

    def test_first_second_of_next_bar(self, live_bot):
        ts = datetime(2025, 1, 1, 9, 35, 0, tzinfo=timezone.utc)
        assert live_bot._get_bar_time(ts) == datetime(2025, 1, 1, 9, 35, 0, tzinfo=timezone.utc)

    def test_strips_seconds_and_microseconds(self, live_bot):
        ts = datetime(2025, 1, 1, 9, 32, 17, 999999, tzinfo=timezone.utc)
        result = live_bot._get_bar_time(ts)
        assert result.second == 0
        assert result.microsecond == 0


# ── find_contract ─────────────────────────────────────────────────────────────

class TestFindContract:
    def test_finds_known_contract(self, live_bot):
        c = live_bot.find_contract("CON.F.US.MNQ.Z25")
        assert c is not None
        assert c["tickSize"] == 0.25

    def test_returns_none_for_unknown(self, live_bot):
        assert live_bot.find_contract("CON.F.US.UNKNOWN") is None

    def test_empty_contracts_list(self, live_bot):
        live_bot.contracts = []
        assert live_bot.find_contract("CON.F.US.MNQ.Z25") is None

    def test_none_contracts_list(self, live_bot):
        live_bot.contracts = None
        with pytest.raises((TypeError, AttributeError)):
            live_bot.find_contract("CON.F.US.MNQ.Z25")


# ── _get_tick_size / _get_tick_value ──────────────────────────────────────────

class TestTickSpecs:
    def test_tick_size_from_contract_details(self, live_bot):
        assert live_bot._get_tick_size() == 0.25

    def test_tick_value_from_contract_details(self, live_bot):
        assert live_bot._get_tick_value() == 0.50

    def test_tick_size_fallback_when_no_contracts(self, live_bot):
        live_bot.contracts = []
        assert live_bot._get_tick_size() == 0.01

    def test_tick_value_fallback_when_no_contracts(self, live_bot):
        live_bot.contracts = []
        assert live_bot._get_tick_value() == 0.50


# ── handle_trade: bar aggregation ─────────────────────────────────────────────

class TestHandleTradeBarAggregation:
    """Ticks should be aggregated into the correct OHLCV bar."""

    def _tick(self, ts_str, price, volume=10):
        return {"timestamp": ts_str, "price": price, "volume": volume}

    def test_first_tick_opens_bar(self, live_bot):
        run(live_bot.handle_trade(self._tick("2025-01-01T09:30:00+00:00", 100.0)))
        assert live_bot.current_bar["open"] == 100.0
        assert live_bot.current_bar["close"] == 100.0

    def test_second_tick_updates_close(self, live_bot):
        run(live_bot.handle_trade(self._tick("2025-01-01T09:30:00+00:00", 100.0)))
        run(live_bot.handle_trade(self._tick("2025-01-01T09:31:00+00:00", 102.0)))
        assert live_bot.current_bar["close"] == 102.0
        assert live_bot.current_bar["open"] == 100.0

    def test_high_updated_on_new_peak(self, live_bot):
        run(live_bot.handle_trade(self._tick("2025-01-01T09:30:00+00:00", 100.0)))
        run(live_bot.handle_trade(self._tick("2025-01-01T09:31:00+00:00", 105.0)))
        assert live_bot.current_bar["high"] == 105.0

    def test_low_updated_on_new_trough(self, live_bot):
        run(live_bot.handle_trade(self._tick("2025-01-01T09:30:00+00:00", 100.0)))
        run(live_bot.handle_trade(self._tick("2025-01-01T09:31:00+00:00", 97.0)))
        assert live_bot.current_bar["low"] == 97.0

    def test_high_not_lowered_by_later_tick(self, live_bot):
        run(live_bot.handle_trade(self._tick("2025-01-01T09:30:00+00:00", 100.0)))
        run(live_bot.handle_trade(self._tick("2025-01-01T09:31:00+00:00", 105.0)))
        run(live_bot.handle_trade(self._tick("2025-01-01T09:32:00+00:00", 98.0)))
        assert live_bot.current_bar["high"] == 105.0

    def test_volume_accumulates(self, live_bot):
        run(live_bot.handle_trade(self._tick("2025-01-01T09:30:00+00:00", 100.0, volume=10)))
        run(live_bot.handle_trade(self._tick("2025-01-01T09:31:00+00:00", 101.0, volume=20)))
        assert live_bot.current_bar["volume"] == 30

    def test_new_bar_time_resets_open(self, live_bot):
        run(live_bot.handle_trade(self._tick("2025-01-01T09:30:00+00:00", 100.0)))
        run(live_bot.handle_trade(self._tick("2025-01-01T09:31:00+00:00", 105.0)))
        # Tick arrives in the *next* bar (09:35)
        run(live_bot.handle_trade(self._tick("2025-01-01T09:35:00+00:00", 102.0)))
        assert live_bot.current_bar["open"] == 102.0

    def test_tick_missing_price_is_ignored(self, live_bot):
        run(live_bot.handle_trade({"timestamp": "2025-01-01T09:30:00+00:00", "volume": 10}))
        assert live_bot.current_bar == {}


# ── handle_trade: exit checking ───────────────────────────────────────────────

class TestHandleTradeExitChecking:
    """When in a position, handle_trade should call _check_exit_conditions."""

    def _put_in_position(self, bot, position_type, entry, stop, target):
        bot.in_position = True
        bot.position_type = position_type
        bot.entry_price = entry
        bot.stop_loss = stop
        bot.profit_target = target
        bot.entry_timestamp = datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc)

    def test_long_stop_hit_clears_position(self, live_bot):
        self._put_in_position(live_bot, "LONG", entry=100, stop=95, target=110)
        run(live_bot.handle_trade({
            "timestamp": "2025-01-01T09:30:00+00:00", "price": 94.0, "volume": 1
        }))
        assert not live_bot.in_position

    def test_long_target_hit_clears_position(self, live_bot):
        self._put_in_position(live_bot, "LONG", entry=100, stop=95, target=110)
        run(live_bot.handle_trade({
            "timestamp": "2025-01-01T09:30:00+00:00", "price": 111.0, "volume": 1
        }))
        assert not live_bot.in_position

    def test_short_stop_hit_clears_position(self, live_bot):
        self._put_in_position(live_bot, "SHORT", entry=100, stop=105, target=90)
        run(live_bot.handle_trade({
            "timestamp": "2025-01-01T09:30:00+00:00", "price": 106.0, "volume": 1
        }))
        assert not live_bot.in_position

    def test_short_target_hit_clears_position(self, live_bot):
        self._put_in_position(live_bot, "SHORT", entry=100, stop=105, target=90)
        run(live_bot.handle_trade({
            "timestamp": "2025-01-01T09:30:00+00:00", "price": 89.0, "volume": 1
        }))
        assert not live_bot.in_position

    def test_price_between_levels_keeps_position(self, live_bot):
        self._put_in_position(live_bot, "LONG", entry=100, stop=95, target=110)
        run(live_bot.handle_trade({
            "timestamp": "2025-01-01T09:30:00+00:00", "price": 102.0, "volume": 1
        }))
        assert live_bot.in_position

    def test_no_position_no_exit_check(self, live_bot):
        """No position — tick should just update the bar, no errors."""
        live_bot.in_position = False
        run(live_bot.handle_trade({
            "timestamp": "2025-01-01T09:30:00+00:00", "price": 100.0, "volume": 1
        }))
        assert not live_bot.in_position


# ── process_tick: input format dispatching ────────────────────────────────────

class TestProcessTick:
    """process_tick handles both list and dict tick formats."""

    def test_dict_format_creates_bar(self, live_bot):
        run(live_bot.process_tick({
            "timestamp": "2025-01-01T09:30:00+00:00", "price": 100.0, "volume": 5
        }))
        assert live_bot.current_bar.get("open") == 100.0

    def test_list_format_creates_bar(self, live_bot):
        run(live_bot.process_tick([
            "CON.F.US.MNQ.Z25",
            [{"timestamp": "2025-01-01T09:30:00+00:00", "price": 100.0, "volume": 5}]
        ]))
        assert live_bot.current_bar.get("open") == 100.0

    def test_list_with_multiple_ticks(self, live_bot):
        run(live_bot.process_tick([
            "CON.F.US.MNQ.Z25",
            [
                {"timestamp": "2025-01-01T09:30:00+00:00", "price": 100.0, "volume": 5},
                {"timestamp": "2025-01-01T09:31:00+00:00", "price": 105.0, "volume": 3},
            ]
        ]))
        assert live_bot.current_bar["high"] == 105.0

    def test_empty_list_payload_ignored(self, live_bot):
        run(live_bot.process_tick([]))
        assert live_bot.current_bar == {}

    def test_malformed_tick_does_not_raise(self, live_bot):
        """Errors inside process_tick are caught — bot should not crash."""
        run(live_bot.process_tick("garbage"))
