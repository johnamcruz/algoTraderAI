"""Shared pytest fixtures for algoTraderAI unit tests."""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.helpers import MockStrategy, ConcreteBot
from bots.simulation_bot import SimulationBot


@pytest.fixture
def mock_strategy():
    return MockStrategy()


@pytest.fixture
def bot(mock_strategy):
    return ConcreteBot(
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


@pytest.fixture
def sim_bot(mock_strategy, tmp_path):
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(
        "time,open,high,low,close,volume\n"
        "1700000000,100,110,90,105,1000\n"
        "1700000060,105,115,95,110,1200\n"
    )
    return SimulationBot(
        csv_path=str(csv_path),
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
