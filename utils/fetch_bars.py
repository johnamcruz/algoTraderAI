#!/usr/bin/env python3
"""
Fetch historical OHLCV bars from the TopstepX History API.

Returns a pandas DataFrame with a timezone-aware (America/New_York) DatetimeIndex
and columns: open, high, low, close, volume — identical to what SimulationBot
expects from a CSV file.
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from utils.bot_utils import BASE_URL, authenticate

# 5-min bars: ~1 440 bars/week (6 trading days × 23 h × 12 bars/h).
# Chunk comfortably under any reasonable API limit.
_CHUNK_DAYS = 7
_LIMIT_PER_CALL = 2000
_TIMEFRAME_UNIT = 2  # unit=2 → minutes (matches trading_bot.py)


def fetch_bars(
    contract_id: str,
    start_date: str,
    end_date: str,
    timeframe_minutes: int,
    username: str,
    api_key: str,
    base_url: str = BASE_URL,
    chunk_days: int = _CHUNK_DAYS,
    limit_per_call: int = _LIMIT_PER_CALL,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars from the TopstepX History API for a date range.

    Paginates in `chunk_days`-wide windows so arbitrarily long ranges work.

    Args:
        contract_id: Full contract ID, e.g. "CON.F.US.MES.M26"
        start_date:  "YYYY-MM-DD" inclusive start
        end_date:    "YYYY-MM-DD" inclusive end
        timeframe_minutes: Bar size in minutes (e.g. 5)
        username:    TopstepX username
        api_key:     TopstepX API key
        base_url:    API base URL
        chunk_days:  Days per API call (default 7)
        limit_per_call: Max bars per API call (default 2000)

    Returns:
        DataFrame with DatetimeIndex (America/New_York) and OHLCV columns.
        Empty DataFrame if no bars are returned.
    """
    token = authenticate(base_url, username, api_key)
    if not token:
        raise RuntimeError("Authentication failed — check username/api_key")

    headers = {"Authorization": f"Bearer {token}"}
    url = f"{base_url}/History/retrieveBars"

    chunk_start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    final_end = (
        datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        + timedelta(days=1)  # inclusive end-date
    )

    all_bars: list[dict] = []

    while chunk_start < final_end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), final_end)

        payload = {
            "contractId": contract_id,
            "live": False,
            "startTime": chunk_start.isoformat().replace("+00:00", "Z"),
            "endTime": chunk_end.isoformat().replace("+00:00", "Z"),
            "unit": _TIMEFRAME_UNIT,
            "unitNumber": timeframe_minutes,
            "limit": limit_per_call,
            "includePartialBar": False,
        }

        logging.debug(f"fetch_bars: {payload['startTime']} → {payload['endTime']}")

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"History/retrieveBars request failed: {exc}") from exc

        bars = resp.json().get("bars", [])
        logging.debug(f"fetch_bars: received {len(bars)} bars")
        all_bars.extend(bars)

        chunk_start = chunk_end

    if not all_bars:
        logging.warning(f"fetch_bars: no bars returned for {contract_id} {start_date}→{end_date}")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    rows = [
        {
            "timestamp": bar["t"],
            "open": bar["o"],
            "high": bar["h"],
            "low": bar["l"],
            "close": bar["c"],
            "volume": bar.get("v", 0),
        }
        for bar in all_bars
    ]

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(
        "America/New_York"
    )
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df.set_index("timestamp", inplace=True)

    logging.info(
        f"fetch_bars: {len(df)} bars fetched for {contract_id} "
        f"({df.index[0]} → {df.index[-1]})"
    )
    return df
