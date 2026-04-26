"""Unit tests for RealTimeBot (trading_bot.py)."""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from tests.helpers import MockStrategy


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def live_bot(mock_strategy):
    """RealTimeBot with SignalRClient patched out."""
    with patch("bots.trading_bot.SignalRClient"):
        from bots.trading_bot import RealTimeBot
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


# ── handle_trade: stale tick discard ─────────────────────────────────────────

class TestStaleTick:
    """Ticks whose bar_time is older than current_bar_time are discarded.

    Scenario: bar_closer_watcher closes the 09:30 bar and advances
    current_bar_time to 09:35. A delayed tick then arrives with a timestamp
    of 09:30:xx — it must be silently dropped to prevent ghost bars.
    """

    def _tick(self, ts_str, price, volume=1):
        return {"timestamp": ts_str, "price": price, "volume": volume}

    def test_stale_tick_does_not_reopen_closed_bar(self, live_bot):
        # Advance to 09:35 bar
        run(live_bot.handle_trade(self._tick("2025-01-01T09:35:00+00:00", 100.0)))
        bar_time_before = live_bot.current_bar_time

        # Stale tick from 09:30 (already closed)
        run(live_bot.handle_trade(self._tick("2025-01-01T09:30:45+00:00", 50.0)))

        assert live_bot.current_bar_time == bar_time_before
        assert live_bot.current_bar["open"] == 100.0

    def test_stale_tick_does_not_corrupt_ohlcv(self, live_bot):
        run(live_bot.handle_trade(self._tick("2025-01-01T09:35:00+00:00", 100.0)))
        run(live_bot.handle_trade(self._tick("2025-01-01T09:36:00+00:00", 102.0)))

        # Stale tick with extreme price should not affect the 09:35 bar
        run(live_bot.handle_trade(self._tick("2025-01-01T09:30:00+00:00", 999.0)))

        assert live_bot.current_bar["high"] == pytest.approx(102.0)
        assert live_bot.current_bar["open"] == pytest.approx(100.0)

    def test_valid_next_bar_tick_is_not_discarded(self, live_bot):
        run(live_bot.handle_trade(self._tick("2025-01-01T09:35:00+00:00", 100.0)))

        # Tick for the NEXT bar (09:40) — should open a new bar, not be dropped
        run(live_bot.handle_trade(self._tick("2025-01-01T09:40:00+00:00", 105.0)))

        assert live_bot.current_bar["open"] == pytest.approx(105.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok(data):
    """Build a mock requests.Response with the given JSON body."""
    m = MagicMock()
    m.raise_for_status.return_value = None
    m.json.return_value = data
    return m


def _place_ok(order_id=2000):
    return _ok({"success": True, "orderId": order_id, "errorCode": 0, "errorMessage": None})


def _search_ok(stop_id=2001, limit_id=2002, contract="CON.F.US.MNQ.Z25", stop_price=95.0, limit_price=110.0):
    return _ok({
        "success": True,
        "orders": [
            {"id": stop_id,  "contractId": contract, "type": 4, "side": 1,
             "stopPrice": stop_price,  "limitPrice": None,        "status": 1},
            {"id": limit_id, "contractId": contract, "type": 1, "side": 1,
             "stopPrice": None,        "limitPrice": limit_price, "status": 1},
        ]
    })


def _cancel_ok():
    return _ok({"success": True, "errorCode": 0, "errorMessage": None})


def _fail(msg="some error"):
    return _ok({"success": False, "errorCode": 1, "errorMessage": msg})


# ── _fetch_stop_bracket_order_id ──────────────────────────────────────────────

class TestFetchStopBracketOrderId:
    """_fetch_stop_bracket_order_id: finds and returns the stop bracket order ID."""

    def test_returns_stop_id_when_found(self, live_bot):
        with patch("bots.trading_bot.requests.post", return_value=_search_ok(stop_id=9001)):
            result = live_bot._fetch_stop_bracket_order_id()
        assert result == 9001

    def test_returns_none_when_no_stop_order(self, live_bot):
        # Only a limit (take-profit) order, no stop
        resp = _ok({
            "success": True,
            "orders": [
                {"id": 9002, "contractId": "CON.F.US.MNQ.Z25", "type": 1,
                 "side": 1, "stopPrice": None, "limitPrice": 110.0, "status": 1}
            ]
        })
        with patch("bots.trading_bot.requests.post", return_value=resp):
            result = live_bot._fetch_stop_bracket_order_id()
        assert result is None

    def test_returns_none_on_api_error(self, live_bot):
        with patch("bots.trading_bot.requests.post", return_value=_fail("searchOpen failed")):
            result = live_bot._fetch_stop_bracket_order_id()
        assert result is None

    def test_returns_none_on_request_exception(self, live_bot):
        with patch("bots.trading_bot.requests.post", side_effect=ConnectionError("timeout")):
            result = live_bot._fetch_stop_bracket_order_id()
        assert result is None

    def test_ignores_stop_orders_from_other_contracts(self, live_bot):
        # Stop order belongs to a different contract — should be filtered out
        resp = _ok({
            "success": True,
            "orders": [
                {"id": 9003, "contractId": "CON.F.US.MES.Z25", "type": 4,
                 "side": 1, "stopPrice": 95.0, "limitPrice": None, "status": 1}
            ]
        })
        with patch("bots.trading_bot.requests.post", return_value=resp):
            result = live_bot._fetch_stop_bracket_order_id()
        assert result is None

    def test_returns_first_stop_when_multiple_present(self, live_bot):
        resp = _ok({
            "success": True,
            "orders": [
                {"id": 9010, "contractId": "CON.F.US.MNQ.Z25", "type": 4,
                 "side": 1, "stopPrice": 95.0, "limitPrice": None, "status": 1},
                {"id": 9011, "contractId": "CON.F.US.MNQ.Z25", "type": 4,
                 "side": 1, "stopPrice": 94.0, "limitPrice": None, "status": 1},
            ]
        })
        with patch("bots.trading_bot.requests.post", return_value=resp):
            result = live_bot._fetch_stop_bracket_order_id()
        assert result == 9010


# ── _on_breakeven_triggered ───────────────────────────────────────────────────

def _modify_ok():
    return _ok({"success": True, "errorCode": 0, "errorMessage": None})


# ── _modify_order ─────────────────────────────────────────────────────────────

class TestModifyOrder:
    """_modify_order: sends Order/modify and returns True/False.

    Reusable primitive for any stop-price update (breakeven, trailing stop, etc).
    """

    def test_posts_to_order_modify_url(self, live_bot):
        with patch("bots.trading_bot.requests.post", return_value=_modify_ok()) as mock_post:
            live_bot._modify_order(order_id=99, stop_price=100.0, size=1)
        url = mock_post.call_args[0][0]
        assert url.endswith("/Order/modify")

    def test_request_body_fields(self, live_bot):
        with patch("bots.trading_bot.requests.post", return_value=_modify_ok()) as mock_post:
            live_bot._modify_order(order_id=42, stop_price=7168.75, size=3)
        body = mock_post.call_args[1]["json"]
        assert body["accountId"]  == live_bot.account
        assert body["orderId"]    == 42
        assert body["stopPrice"]  == pytest.approx(7168.75)
        assert body["size"]       == 3
        assert body["limitPrice"] is None
        assert body["trailPrice"] is None

    def test_returns_true_on_success(self, live_bot):
        with patch("bots.trading_bot.requests.post", return_value=_modify_ok()):
            assert live_bot._modify_order(99, 100.0, 1) is True

    def test_returns_false_on_api_failure(self, live_bot):
        with patch("bots.trading_bot.requests.post", return_value=_fail("invalid stop price")):
            assert live_bot._modify_order(99, 100.0, 1) is False

    def test_returns_false_on_exception(self, live_bot):
        with patch("bots.trading_bot.requests.post", side_effect=ConnectionError("timeout")):
            assert live_bot._modify_order(99, 100.0, 1) is False

    def test_http_error_is_handled_gracefully(self, live_bot):
        import requests as req
        err_response = MagicMock()
        err_response.raise_for_status.side_effect = req.HTTPError("503")
        with patch("bots.trading_bot.requests.post", return_value=err_response):
            assert live_bot._modify_order(99, 100.0, 1) is False   # no crash

    def test_fractional_stop_price_preserved(self, live_bot):
        with patch("bots.trading_bot.requests.post", return_value=_modify_ok()) as mock_post:
            live_bot._modify_order(99, 7168.75, 1)
        assert mock_post.call_args[1]["json"]["stopPrice"] == pytest.approx(7168.75)

    def test_large_size_passed_correctly(self, live_bot):
        with patch("bots.trading_bot.requests.post", return_value=_modify_ok()) as mock_post:
            live_bot._modify_order(99, 100.0, 15)
        assert mock_post.call_args[1]["json"]["size"] == 15


# ── _on_breakeven_triggered ───────────────────────────────────────────────────

class TestOnBreakevenTriggered:
    """_on_breakeven_triggered: delegates to _modify_order with entry price.

    Guard logic (no bracket ID) lives here; API details are tested in TestModifyOrder.
    """

    def _setup(self, bot, bracket_id=5000, entry=100.0, size=2,
               position_type="LONG"):
        bot.in_position = True
        bot.position_type = position_type
        bot.entry_price = entry
        bot.stop_loss = entry
        bot.stop_bracket_order_id = bracket_id
        bot.position_size = size

    def test_no_op_when_no_bracket_id(self, live_bot):
        live_bot.stop_bracket_order_id = None
        live_bot.entry_price = 100.0
        with patch("bots.trading_bot.requests.post") as mock_post:
            run(live_bot._on_breakeven_triggered())
        mock_post.assert_not_called()

    def test_calls_modify_order_with_entry_price(self, live_bot):
        self._setup(live_bot, bracket_id=5001, entry=100.0, size=2)
        with patch.object(live_bot, "_modify_order", return_value=True) as mock_mod:
            run(live_bot._on_breakeven_triggered())
        mock_mod.assert_called_once_with(
            order_id=5001, stop_price=100.0, size=2
        )

    def test_bracket_id_unchanged_after_modify(self, live_bot):
        self._setup(live_bot, bracket_id=5002, entry=150.0)
        with patch.object(live_bot, "_modify_order", return_value=True):
            run(live_bot._on_breakeven_triggered())
        assert live_bot.stop_bracket_order_id == 5002

    def test_no_op_skips_modify_entirely(self, live_bot):
        live_bot.stop_bracket_order_id = None
        live_bot.entry_price = 100.0
        with patch.object(live_bot, "_modify_order") as mock_mod:
            run(live_bot._on_breakeven_triggered())
        mock_mod.assert_not_called()

    def test_modify_failure_does_not_raise(self, live_bot):
        self._setup(live_bot, bracket_id=5003, entry=100.0)
        with patch.object(live_bot, "_modify_order", return_value=False):
            run(live_bot._on_breakeven_triggered())   # must not raise

    def test_modify_failure_rolls_back_breakeven_state(self, live_bot):
        # If API fails, breakeven_set must be reset so the next tick retries
        self._setup(live_bot, bracket_id=5003, entry=100.0)
        live_bot.breakeven_set = True
        live_bot._pre_breakeven_stop = 95.0
        with patch.object(live_bot, "_modify_order", return_value=False):
            run(live_bot._on_breakeven_triggered())
        assert live_bot.breakeven_set is False
        assert live_bot.stop_loss == pytest.approx(95.0)

    def test_no_bracket_id_rolls_back_breakeven_state(self, live_bot):
        # No bracket ID — same rollback so next tick retries
        live_bot.stop_bracket_order_id = None
        live_bot.entry_price = 100.0
        live_bot.stop_loss = 100.0
        live_bot.breakeven_set = True
        live_bot._pre_breakeven_stop = 95.0
        run(live_bot._on_breakeven_triggered())
        assert live_bot.breakeven_set is False

    def test_works_for_short_position(self, live_bot):
        self._setup(live_bot, bracket_id=5004, entry=200.0, size=1,
                    position_type="SHORT")
        with patch.object(live_bot, "_modify_order", return_value=True) as mock_mod:
            run(live_bot._on_breakeven_triggered())
        mock_mod.assert_called_once_with(
            order_id=5004, stop_price=200.0, size=1
        )

    def test_fractional_entry_price_forwarded(self, live_bot):
        self._setup(live_bot, bracket_id=5005, entry=7168.75, size=3)
        with patch.object(live_bot, "_modify_order", return_value=True) as mock_mod:
            run(live_bot._on_breakeven_triggered())
        _, kwargs = mock_mod.call_args
        assert kwargs["stop_price"] == pytest.approx(7168.75)

    def test_does_not_raise_when_position_size_is_none(self, live_bot):
        self._setup(live_bot, bracket_id=5006, entry=100.0, size=None)
        with patch.object(live_bot, "_modify_order", return_value=True):
            run(live_bot._on_breakeven_triggered())   # must not raise


# ── _place_order: position state ──────────────────────────────────────────────

class TestPlaceOrderSetsState:
    """_place_order sets position state and fetches stop bracket ID on success."""

    def _run_place(self, bot, side=0, close=100.0, stop=95.0, target=110.0, size=2):
        with patch("bots.trading_bot.requests.post", side_effect=[_place_ok(2000), _search_ok(stop_id=2001)]), \
             patch("asyncio.sleep", new=AsyncMock()):
            return run(bot._place_order(
                side=side, close_price=close, stop_loss=stop,
                profit_target=target, stop_ticks=-20, take_profit_ticks=40, size=size
            ))

    def test_sets_in_position_true(self, live_bot):
        self._run_place(live_bot)
        assert live_bot.in_position is True

    def test_sets_long_position_type(self, live_bot):
        self._run_place(live_bot, side=0)
        assert live_bot.position_type == "LONG"

    def test_sets_short_position_type(self, live_bot):
        self._run_place(live_bot, side=1)
        assert live_bot.position_type == "SHORT"

    def test_sets_entry_price_to_close_price(self, live_bot):
        self._run_place(live_bot, close=123.45)
        assert live_bot.entry_price == pytest.approx(123.45)

    def test_sets_stop_loss(self, live_bot):
        self._run_place(live_bot, stop=95.0)
        assert live_bot.stop_loss == pytest.approx(95.0)

    def test_sets_profit_target(self, live_bot):
        self._run_place(live_bot, target=110.0)
        assert live_bot.profit_target == pytest.approx(110.0)

    def test_sets_position_size(self, live_bot):
        self._run_place(live_bot, size=3)
        assert live_bot.position_size == 3

    def test_stores_stop_bracket_id(self, live_bot):
        self._run_place(live_bot)
        assert live_bot.stop_bracket_order_id == 2001

    def test_returns_order_id(self, live_bot):
        result = self._run_place(live_bot)
        assert result == 2000

    def test_no_state_set_on_api_failure(self, live_bot):
        with patch("bots.trading_bot.requests.post", return_value=_fail("order rejected")), \
             patch("asyncio.sleep", new=AsyncMock()):
            run(live_bot._place_order(
                side=0, close_price=100.0, stop_loss=95.0,
                profit_target=110.0, stop_ticks=-20, take_profit_ticks=40, size=1
            ))
        assert live_bot.in_position is False
        assert live_bot.entry_price is None
        assert live_bot.stop_bracket_order_id is None

    def test_returns_none_on_api_failure(self, live_bot):
        with patch("bots.trading_bot.requests.post", return_value=_fail()), \
             patch("asyncio.sleep", new=AsyncMock()):
            result = run(live_bot._place_order(
                side=0, close_price=100.0, stop_loss=95.0,
                profit_target=110.0, stop_ticks=-20, take_profit_ticks=40, size=1
            ))
        assert result is None
