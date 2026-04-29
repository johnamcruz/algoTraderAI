"""
Unit tests for utils/fetch_bars.py

Tests cover:
  - Single-chunk fetch (short date range)
  - Multi-chunk pagination across date boundaries
  - Deduplication of bars returned by overlapping chunks
  - Empty API response returns empty DataFrame with correct columns
  - API authentication failure raises RuntimeError
  - HTTP error raises RuntimeError
  - DataFrame schema: correct index name, columns, timezone
  - Bars are sorted ascending by timestamp
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from utils.fetch_bars import fetch_bars


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_bar(dt_str: str, price: float = 100.0) -> dict:
    """Return a minimal bar dict as the API would return it."""
    return {
        "t": dt_str,
        "o": price,
        "h": price + 1,
        "l": price - 1,
        "c": price,
        "v": 500,
    }


def _api_response(bars: list) -> MagicMock:
    mock = MagicMock()
    mock.json.return_value = {"bars": bars}
    mock.raise_for_status.return_value = None
    return mock


# ── Tests ─────────────────────────────────────────────────────────────────────

@patch("utils.fetch_bars.authenticate", return_value="mock-token")
@patch("utils.fetch_bars.requests.post")
def test_single_chunk_returns_dataframe(mock_post, mock_auth):
    """A short range (≤ chunk_days) makes one API call and returns a DataFrame."""
    bars = [
        _make_bar("2026-04-01T10:00:00Z", 100),
        _make_bar("2026-04-01T10:05:00Z", 101),
        _make_bar("2026-04-01T10:10:00Z", 102),
    ]
    mock_post.return_value = _api_response(bars)

    df = fetch_bars(
        contract_id="CON.F.US.MES.M26",
        start_date="2026-04-01",
        end_date="2026-04-03",
        timeframe_minutes=5,
        username="user",
        api_key="key",
        chunk_days=7,
    )

    assert mock_post.call_count == 1
    assert len(df) == 3
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "timestamp"


@patch("utils.fetch_bars.authenticate", return_value="mock-token")
@patch("utils.fetch_bars.requests.post")
def test_multi_chunk_pagination(mock_post, mock_auth):
    """A range spanning multiple chunks issues one call per chunk."""
    chunk1 = [_make_bar("2026-04-01T10:00:00Z", 100)]
    chunk2 = [_make_bar("2026-04-08T10:00:00Z", 200)]
    chunk3 = [_make_bar("2026-04-15T10:00:00Z", 300)]
    mock_post.side_effect = [
        _api_response(chunk1),
        _api_response(chunk2),
        _api_response(chunk3),
    ]

    df = fetch_bars(
        contract_id="CON.F.US.MES.M26",
        start_date="2026-04-01",
        end_date="2026-04-21",
        timeframe_minutes=5,
        username="user",
        api_key="key",
        chunk_days=7,
    )

    assert mock_post.call_count == 3
    assert len(df) == 3


@patch("utils.fetch_bars.authenticate", return_value="mock-token")
@patch("utils.fetch_bars.requests.post")
def test_deduplication(mock_post, mock_auth):
    """Duplicate timestamps across chunks are dropped."""
    bar = _make_bar("2026-04-07T23:55:00Z", 100)
    mock_post.side_effect = [
        _api_response([bar, bar]),  # same bar twice in one chunk
        _api_response([]),
    ]

    df = fetch_bars(
        contract_id="CON.F.US.MES.M26",
        start_date="2026-04-01",
        end_date="2026-04-14",
        timeframe_minutes=5,
        username="user",
        api_key="key",
        chunk_days=7,
    )

    assert len(df) == 1


@patch("utils.fetch_bars.authenticate", return_value="mock-token")
@patch("utils.fetch_bars.requests.post")
def test_empty_response_returns_empty_dataframe(mock_post, mock_auth):
    """When the API returns no bars, an empty DataFrame with correct columns is returned."""
    mock_post.return_value = _api_response([])

    df = fetch_bars(
        contract_id="CON.F.US.MES.M26",
        start_date="2026-04-01",
        end_date="2026-04-03",
        timeframe_minutes=5,
        username="user",
        api_key="key",
    )

    assert df.empty
    assert set(["open", "high", "low", "close", "volume"]).issubset(set(df.columns))


@patch("utils.fetch_bars.authenticate", return_value=None)
def test_auth_failure_raises(mock_auth):
    """Authentication failure raises RuntimeError immediately."""
    with pytest.raises(RuntimeError, match="Authentication failed"):
        fetch_bars(
            contract_id="CON.F.US.MES.M26",
            start_date="2026-04-01",
            end_date="2026-04-03",
            timeframe_minutes=5,
            username="bad",
            api_key="bad",
        )


@patch("utils.fetch_bars.authenticate", return_value="mock-token")
@patch("utils.fetch_bars.requests.post")
def test_http_error_raises(mock_post, mock_auth):
    """HTTP error from the API raises RuntimeError."""
    import requests as req
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = req.HTTPError("500 Server Error")
    mock_post.return_value = mock_resp

    with pytest.raises(RuntimeError, match="History/retrieveBars request failed"):
        fetch_bars(
            contract_id="CON.F.US.MES.M26",
            start_date="2026-04-01",
            end_date="2026-04-03",
            timeframe_minutes=5,
            username="user",
            api_key="key",
        )


@patch("utils.fetch_bars.authenticate", return_value="mock-token")
@patch("utils.fetch_bars.requests.post")
def test_timezone_is_eastern(mock_post, mock_auth):
    """Returned DataFrame index is in America/New_York timezone."""
    bars = [_make_bar("2026-04-01T14:00:00Z")]
    mock_post.return_value = _api_response(bars)

    df = fetch_bars(
        contract_id="CON.F.US.MES.M26",
        start_date="2026-04-01",
        end_date="2026-04-01",
        timeframe_minutes=5,
        username="user",
        api_key="key",
    )

    assert str(df.index.tz) == "America/New_York"


@patch("utils.fetch_bars.authenticate", return_value="mock-token")
@patch("utils.fetch_bars.requests.post")
def test_bars_sorted_ascending(mock_post, mock_auth):
    """Bars are returned in ascending timestamp order regardless of API order."""
    bars = [
        _make_bar("2026-04-01T10:10:00Z", 102),
        _make_bar("2026-04-01T10:00:00Z", 100),
        _make_bar("2026-04-01T10:05:00Z", 101),
    ]
    mock_post.return_value = _api_response(bars)

    df = fetch_bars(
        contract_id="CON.F.US.MES.M26",
        start_date="2026-04-01",
        end_date="2026-04-01",
        timeframe_minutes=5,
        username="user",
        api_key="key",
    )

    assert list(df["close"]) == [100.0, 101.0, 102.0]


@patch("utils.fetch_bars.authenticate", return_value="mock-token")
@patch("utils.fetch_bars.requests.post")
def test_volume_defaults_to_zero_when_missing(mock_post, mock_auth):
    """Bars without a 'v' field get volume=0."""
    bar = {"t": "2026-04-01T10:00:00Z", "o": 100, "h": 101, "l": 99, "c": 100}
    mock_post.return_value = _api_response([bar])

    df = fetch_bars(
        contract_id="CON.F.US.MES.M26",
        start_date="2026-04-01",
        end_date="2026-04-01",
        timeframe_minutes=5,
        username="user",
        api_key="key",
    )

    assert df["volume"].iloc[0] == 0


@patch("utils.fetch_bars.authenticate", return_value="mock-token")
@patch("utils.fetch_bars.requests.post")
def test_correct_payload_sent(mock_post, mock_auth):
    """Verify the API payload contains the expected fields."""
    mock_post.return_value = _api_response([])

    fetch_bars(
        contract_id="CON.F.US.MNQ.M26",
        start_date="2026-04-01",
        end_date="2026-04-03",
        timeframe_minutes=5,
        username="user",
        api_key="key",
    )

    payload = mock_post.call_args.kwargs["json"]
    assert payload["contractId"] == "CON.F.US.MNQ.M26"
    assert payload["unitNumber"] == 5
    assert payload["unit"] == 2
    assert payload["live"] is False
    assert payload["includePartialBar"] is False
