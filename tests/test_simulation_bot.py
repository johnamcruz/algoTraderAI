"""Unit tests for SimulationBot."""

import pytest
import pandas as pd
from simulation_bot import SimulationBot
from tests.helpers import MockStrategy


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
            adx_thresh=20.0,
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
            adx_thresh=20.0,
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
