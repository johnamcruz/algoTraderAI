"""Shared test helpers: concrete stubs for abstract base classes."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot_base import TradingBot


class MockStrategy:
    """Minimal stub satisfying BaseStrategy's interface."""

    def get_sequence_length(self):
        return 64

    def get_feature_columns(self):
        return ["open", "high", "low", "close", "volume"]

    def add_features(self, df):
        return df

    def validate_features(self, df):
        return True

    def predict(self, df):
        return (1, 0.85)

    def should_enter_trade(self, prediction, confidence, bar, entry_conf, adx_thresh):
        return (True, "LONG")

    def get_stop_target_pts(self, df, direction, entry_price):
        return (None, None)

    def on_trade_exit(self, reason):
        pass

    def set_contract_symbol(self, symbol):
        pass

    def load_model(self):
        pass

    def load_scaler(self):
        pass


class ConcreteBot(TradingBot):
    """Concrete TradingBot for unit testing (MNQ specs, $0.50/tick, 0.25 tick size)."""

    def _get_tick_size(self):
        return 0.25

    def _get_tick_value(self):
        return 0.50

    async def _has_existing_position(self):
        return False

    async def _place_order(self, side, close_price, stop_loss, profit_target,
                           stop_ticks, take_profit_ticks, size=1):
        pass

    async def run(self):
        pass
