"""Unit tests for bot_utils helpers."""

import pytest
from bot_utils import parse_future_symbol, TICK_VALUES, TICK_SIZES


class TestParseFutureSymbol:
    """parse_future_symbol: extract base symbol from contract name."""

    # Micro → Mini mapping
    def test_mnq_maps_to_nq(self):
        assert parse_future_symbol("MNQZ5") == "NQ"

    def test_mes_maps_to_es(self):
        assert parse_future_symbol("MESZ5") == "ES"

    def test_mgc_maps_to_gc(self):
        assert parse_future_symbol("MGCZ5") == "GC"

    def test_mcl_maps_to_cl(self):
        assert parse_future_symbol("MCLZ5") == "CL"

    def test_mbt_maps_to_btc(self):
        assert parse_future_symbol("MBTZ5") == "BTC"

    # Standard (non-micro) symbols
    def test_nq_returns_nq(self):
        assert parse_future_symbol("NQZ5") == "NQ"

    def test_es_returns_es(self):
        assert parse_future_symbol("ESZ5") == "ES"

    def test_gc_returns_gc(self):
        assert parse_future_symbol("GCZ5") == "GC"

    # Edge cases
    def test_none_returns_none(self):
        assert parse_future_symbol(None) is None

    def test_empty_string_returns_none(self):
        assert parse_future_symbol("") is None

    def test_lowercase_input_normalised(self):
        assert parse_future_symbol("mnqz5") == "NQ"

    def test_dotted_contract_id_format(self):
        # Some brokers prefix with "CON.F.US." — last segment is the symbol
        assert parse_future_symbol("CON.F.US.MNQZ5") == "NQ"

    def test_no_expiry_digit_returns_symbol_as_is(self):
        # No digit → treated as bare symbol (no month stripping)
        result = parse_future_symbol("BTC")
        # "BTC" has no digit, base_symbol = "BTC", not in micro map → returns "BTC"
        assert result == "BTC"

    def test_different_month_codes(self):
        # Test a variety of month codes to ensure stripping works
        for month in "FGHJKMNQUVXZ":
            assert parse_future_symbol(f"NQ{month}5") == "NQ"


class TestTickConstants:
    """Sanity checks on TICK_VALUES and TICK_SIZES dictionaries."""

    def test_mnq_tick_value(self):
        assert TICK_VALUES["MNQ"] == 0.50

    def test_nq_tick_value(self):
        assert TICK_VALUES["NQ"] == 5.00

    def test_es_tick_value(self):
        assert TICK_VALUES["ES"] == 12.50

    def test_mnq_tick_size(self):
        assert TICK_SIZES["MNQ"] == 0.25

    def test_enq_matches_nq(self):
        # Contract-ID code ENQ should equal the NQ tick value
        assert TICK_VALUES["ENQ"] == TICK_VALUES["NQ"]
        assert TICK_SIZES["ENQ"] == TICK_SIZES["NQ"]

    def test_ep_matches_es(self):
        assert TICK_VALUES["EP"] == TICK_VALUES["ES"]
        assert TICK_SIZES["EP"] == TICK_SIZES["ES"]

    def test_all_tick_values_positive(self):
        for symbol, value in TICK_VALUES.items():
            assert value > 0, f"TICK_VALUES[{symbol!r}] must be positive"

    def test_all_tick_sizes_positive(self):
        for symbol, size in TICK_SIZES.items():
            assert size > 0, f"TICK_SIZES[{symbol!r}] must be positive"

    def test_tick_values_and_sizes_same_keys(self):
        assert set(TICK_VALUES.keys()) == set(TICK_SIZES.keys())
