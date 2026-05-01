#!/usr/bin/env python3
"""
CISD+OTE Strategy v7.0 — FFM Hybrid Transformer

Key differences from v5.1:
  - seq_len: 64 → 96
  - FFM features: 42 custom → 67 via futures_foundation.derive_features()
    Market context (HTF trend, structure, volume, session, EMA, etc.) that lived
    in v5.1's 28-feature CISD vector is now encoded by the backbone sequence.
  - CISD features: 28 → 10 (zone geometry + trade mechanics only)
  - New ONNX input: candle_types (int64 [B, seq_len], 0–5)
  - Input renamed: cisd_features → strategy_features
  - Output: signal_logits (raw logits); confidence is max(softmax) — already computed
  - No session filter: model self-regulates via in_optimal_session CISD feature

Walk-forward thresholds (confidence = max(softmax)):
  0.90 = conservative  |  0.80 = moderate  |  0.70 = aggressive
  See cisd_ote_hybrid_metadata.json for per-fold precision at each threshold.

Requirements:
  pip install onnxruntime
  pip install futures-foundation  (or add futures_foundation/ to PYTHONPATH)

Timestamps must be in Eastern Time for session features to match training.
"""

import logging
import numpy as np
import pandas as pd
from collections import deque
from typing import List, Tuple, Optional

from strategies.ffm_strategy_base import FFMStrategyBase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ── CISD detection constants (must match training in cisd_ote.py) ───────────
SWING_PERIOD        = 6
TOLERANCE           = 0.70
EXPIRY_BARS         = 50
LIQUIDITY_LOOKBACK  = 10
ZONE_MAX_BARS       = 40
FIB_1               = 0.618
FIB_2               = 0.786
HTF_RANGE_BARS      = 96
DISP_BODY_RATIO_MIN = 0.50
DISP_CLOSE_STR_MIN  = 0.60

# ── Risk normalization (must match training) ────────────────────────────────
MAX_RISK_DOLLARS = 300.0
POINT_VALUES = {
    'ES': 50.0, 'NQ': 20.0, 'RTY': 10.0, 'YM': 5.0, 'GC': 100.0,
    'MES': 5.0, 'MNQ': 2.0, 'MRTY': 5.0, 'MYM': 0.50, 'MGC': 10.0,
}

# ── Session constants ────────────────────────────────────────────────────────
OPTIMAL_START_HOUR = 9    # in_optimal_session feature: 09:00–11:00 ET
OPTIMAL_END_HOUR   = 11


class CISDOTEStrategyV7(FFMStrategyBase):
    """
    CISD+OTE Hybrid Strategy using FFM Transformer backbone (v7.0).

    Architecture:
      FFMBackbone(256-dim, 67 features × 96 bars) + CISDProjection(10 features)
      → signal_logits [B, 2]       apply softmax for signal prob
        risk_predictions [B, 1]
        confidence [B]             max(softmax) — use this for entry_conf

    CISD features (10):
      zone geometry (0–3), direction (4), setup quality (5–6),
      entry timing (7), risk sizing (8), session flag (9)
    """

    def __init__(
        self,
        model_path: str,
        contract_symbol: str,
        min_risk_rr: float = 2.0,
    ):
        super().__init__(model_path, contract_symbol, min_risk_rr,
                         strategy_tag="CISD+OTE v7")

        # CISD zone state
        self._active_zones:    deque = deque(maxlen=20)
        self._pivot_highs:     deque = deque(maxlen=200)
        self._pivot_lows:      deque = deque(maxlen=200)
        self._last_wicked_high: int  = -999
        self._last_wicked_low:  int  = -999
        self._bear_pots:       deque = deque(maxlen=20)
        self._bull_pots:       deque = deque(maxlen=20)

        self._latest_cisd_features: Optional[np.ndarray] = None
        self._latest_zone_bullish:  float                 = 0.0

        logging.info("=" * 65)
        logging.info("🎯 CISD+OTE Strategy v7.0 — FFM Hybrid Transformer")
        logging.info("=" * 65)
        logging.info(f"  seq_len=96 | FFM features=67 | CISD features=10")
        logging.info(f"  Optimal session feature: {OPTIMAL_START_HOUR}–{OPTIMAL_END_HOUR} ET")
        logging.info(f"  Recommended threshold: 0.80 (moderate)")
        logging.info(f"  Warmup: ~300 bars")
        if min_risk_rr > 0.0:
            logging.info(f"  RR gate: block when predicted_rr < {min_risk_rr}")
        logging.info("=" * 65)

    # ── BaseStrategy interface ────────────────────────────────────────────────

    def get_warmup_length(self) -> int:
        return 200

    @property
    def active_zone_count(self) -> int:
        return len(self._active_zones)

    # ── FFMStrategyBase abstract hooks ────────────────────────────────────────

    def _is_ready_to_predict(self) -> bool:
        return self._latest_cisd_features is not None

    def _get_strategy_features(self) -> np.ndarray:
        return self._latest_cisd_features

    def _get_signal_direction(self) -> int:
        if self._latest_zone_bullish > 0:
            return 1
        if self._latest_zone_bullish < 0:
            return 2
        return 0

    def _after_new_bar(self, df: pd.DataFrame, bar_idx: int) -> None:
        self._latest_cisd_features = self._build_cisd_feature_vector(df, bar_idx)

    def _build_signal_meta(self, confidence: float) -> dict:
        cisd = self._latest_cisd_features
        return {
            'confidence':          round(confidence, 4),
            'risk_rr':             round(self._latest_risk_rr, 4),
            'zone_is_bullish':     round(float(cisd[4]), 4) if cisd is not None else 0.0,
            'had_liquidity_sweep': round(float(cisd[6]), 4) if cisd is not None else 0.0,
            'entry_distance_pct':  round(float(cisd[7]), 4) if cisd is not None else 0.0,
            'in_optimal_session':  round(float(cisd[9]), 4) if cisd is not None else 0.0,
        }

    # ── Incremental bar processing ────────────────────────────────────────────

    def _on_new_bar(self, df: pd.DataFrame, bar_idx: int) -> None:
        self._update_cisd_detector(df, bar_idx)

    # ── Bot hooks ────────────────────────────────────────────────────────────

    def on_trade_exit(self, reason: str):
        if reason == 'STOP_LOSS':
            logging.info(f"🔴 Stop loss — clearing {len(self._active_zones)} zone(s)")
            self._active_zones.clear()
            self._latest_cisd_features = None
            self._latest_zone_bullish  = 0.0

    def get_stop_target_pts(
        self,
        df: pd.DataFrame,
        direction: str,
        entry_price: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        if not self._active_zones:
            return None, None

        nearest = min(
            self._active_zones,
            key=lambda z: abs(entry_price - (z['fib_top'] + z['fib_bot']) / 2.0)
        )
        ft = nearest['fib_top']
        fb = nearest['fib_bot']

        stop_pts = abs(entry_price - fb) if direction == 'LONG' else abs(ft - entry_price)
        if stop_pts <= 0:
            return None, None

        raw_rr = self._latest_risk_rr
        if raw_rr >= 2.0:
            rr = int(raw_rr)
        else:
            return None, None

        target_pts = stop_pts * rr
        logging.info(
            f"  CISD stop/target | dir={direction} entry={entry_price:.2f} "
            f"zone=[{fb:.2f}–{ft:.2f}] stop={stop_pts:.2f}pts "
            f"target={target_pts:.2f}pts (predicted_rr={raw_rr:.2f} → tier={rr}R)"
        )
        return stop_pts, target_pts

    # ── Incremental CISD Zone Detector ────────────────────────────────────────

    def _update_cisd_detector(self, df: pd.DataFrame, abs_bar: int):
        """
        Incremental CISD zone detection using absolute bar indices for stored state.
        Same logic as v5.1 with updated displacement constants.
        """
        n   = len(df)
        bar = n - 1

        if bar < 1:
            return

        o_arr = df['open'].values
        h_arr = df['high'].values
        l_arr = df['low'].values
        c_arr = df['close'].values

        def to_rel(abs_idx):
            return max(0, n - 1 - abs_bar + abs_idx)

        # Track sweep (wick-throughs)
        new_ph = []
        for ph_price, ph_abs in self._pivot_highs:
            if (abs_bar - ph_abs) >= EXPIRY_BARS:
                continue
            if h_arr[bar] >= ph_price:
                self._last_wicked_high = abs_bar
            else:
                new_ph.append((ph_price, ph_abs))
        self._pivot_highs = deque(new_ph, maxlen=200)

        new_pl = []
        for pl_price, pl_abs in self._pivot_lows:
            if (abs_bar - pl_abs) >= EXPIRY_BARS:
                continue
            if l_arr[bar] <= pl_price:
                self._last_wicked_low = abs_bar
            else:
                new_pl.append((pl_price, pl_abs))
        self._pivot_lows = deque(new_pl, maxlen=200)

        # Pivot confirmation
        confirm_rel = bar - SWING_PERIOD
        confirm_abs = abs_bar - SWING_PERIOD
        if confirm_rel >= 1:
            window = h_arr[max(0, confirm_rel - SWING_PERIOD):
                          min(n, confirm_rel + SWING_PERIOD + 1)]
            if len(window) == 2 * SWING_PERIOD + 1:
                center = h_arr[confirm_rel]
                if center == window.max() and (window == center).sum() == 1:
                    self._pivot_highs.append((center, confirm_abs))

            window = l_arr[max(0, confirm_rel - SWING_PERIOD):
                          min(n, confirm_rel + SWING_PERIOD + 1)]
            if len(window) == 2 * SWING_PERIOD + 1:
                center = l_arr[confirm_rel]
                if center == window.min() and (window == center).sum() == 1:
                    self._pivot_lows.append((center, confirm_abs))

        # Track potential CISD candles
        if c_arr[bar-1] < o_arr[bar-1] and c_arr[bar] > o_arr[bar]:
            self._bear_pots.append((o_arr[bar], abs_bar))
        if c_arr[bar-1] > o_arr[bar-1] and c_arr[bar] < o_arr[bar]:
            self._bull_pots.append((o_arr[bar], abs_bar))

        self._bear_pots = deque(
            [(p, b) for p, b in self._bear_pots if abs_bar - b < EXPIRY_BARS],
            maxlen=20)
        self._bull_pots = deque(
            [(p, b) for p, b in self._bull_pots if abs_bar - b < EXPIRY_BARS],
            maxlen=20)

        # P/D midpoint
        rng_h  = h_arr[max(0, bar - HTF_RANGE_BARS):bar].max() if bar > 0 else h_arr[bar]
        rng_l  = l_arr[max(0, bar - HTF_RANGE_BARS):bar].min() if bar > 0 else l_arr[bar]
        pd_mid = (rng_h + rng_l) / 2.0

        # Bearish CISD check
        cisd_zone = None
        cisd_dir  = None
        new_bear  = deque(maxlen=20)
        for pot_price, pot_abs in self._bear_pots:
            pot_rel = to_rel(pot_abs)
            if c_arr[bar] < pot_price:
                highest_c = c_arr[pot_rel:bar+1].max()
                top_level = 0.0
                idx = pot_rel + 1
                while idx < bar and c_arr[idx] < o_arr[idx]:
                    top_level = o_arr[idx]; idx += 1
                if top_level > 0 and (top_level - pot_price) > 0:
                    ratio = (highest_c - pot_price) / (top_level - pot_price)
                    if ratio > TOLERANCE:
                        full_range = h_arr[bar] - l_arr[bar]
                        body = abs(c_arr[bar] - o_arr[bar])
                        br = body / full_range if full_range > 0 else 0.0
                        cs = (h_arr[bar] - c_arr[bar]) / full_range if full_range > 0 else 0.0
                        if br >= DISP_BODY_RATIO_MIN and cs >= DISP_CLOSE_STR_MIN:
                            in_premium = c_arr[bar] > pd_mid
                            had_sweep  = (abs_bar - self._last_wicked_high) <= LIQUIDITY_LOOKBACK
                            if in_premium or had_sweep:
                                h_max = h_arr[pot_rel:bar+1].max()
                                diff  = h_max - l_arr[bar]
                                ft = h_max - diff * FIB_1
                                fb = h_max - diff * FIB_2
                                fib_top = max(ft, fb); fib_bot = min(ft, fb)
                                if fib_top > fib_bot:
                                    cisd_zone = {
                                        'is_bullish': False, 'fib_top': fib_top,
                                        'fib_bot': fib_bot, 'created_bar': abs_bar,
                                        'had_sweep': had_sweep,
                                        'disp_strength': float(ratio),
                                        'signal_fired': False, 'entered_zone': False,
                                    }
                                    cisd_dir = 'BEAR'
                            break
            else:
                new_bear.append((pot_price, pot_abs))
        self._bear_pots = new_bear

        # Bullish CISD check
        new_bull = deque(maxlen=20)
        for pot_price, pot_abs in self._bull_pots:
            pot_rel = to_rel(pot_abs)
            if c_arr[bar] > pot_price and cisd_dir is None:
                lowest_c     = c_arr[pot_rel:bar+1].min()
                bottom_level = 0.0
                idx = pot_rel + 1
                while idx < bar and c_arr[idx] > o_arr[idx]:
                    bottom_level = o_arr[idx]; idx += 1
                if bottom_level > 0 and (pot_price - bottom_level) > 0:
                    ratio = (pot_price - lowest_c) / (pot_price - bottom_level)
                    if ratio > TOLERANCE:
                        full_range = h_arr[bar] - l_arr[bar]
                        body = abs(c_arr[bar] - o_arr[bar])
                        br = body / full_range if full_range > 0 else 0.0
                        cs = (c_arr[bar] - l_arr[bar]) / full_range if full_range > 0 else 0.0
                        if br >= DISP_BODY_RATIO_MIN and cs >= DISP_CLOSE_STR_MIN:
                            in_discount = c_arr[bar] <= pd_mid
                            had_sweep   = (abs_bar - self._last_wicked_low) <= LIQUIDITY_LOOKBACK
                            if in_discount or had_sweep:
                                l_min = l_arr[pot_rel:bar+1].min()
                                diff  = h_arr[bar] - l_min
                                ft = l_min + diff * FIB_1
                                fb = l_min + diff * FIB_2
                                fib_top = max(ft, fb); fib_bot = min(ft, fb)
                                if fib_top > fib_bot:
                                    cisd_zone = {
                                        'is_bullish': True, 'fib_top': fib_top,
                                        'fib_bot': fib_bot, 'created_bar': abs_bar,
                                        'had_sweep': had_sweep,
                                        'disp_strength': float(ratio),
                                        'signal_fired': False, 'entered_zone': False,
                                    }
                                    cisd_dir = 'BULL'
                            break
            else:
                new_bull.append((pot_price, pot_abs))
        self._bull_pots = new_bull

        if cisd_zone is not None:
            self._active_zones.appendleft(cisd_zone)
            logging.info(
                f"🟢 New CISD zone: {'BULL' if cisd_zone['is_bullish'] else 'BEAR'} | "
                f"top={cisd_zone['fib_top']:.2f} bot={cisd_zone['fib_bot']:.2f} | "
                f"total active={len(self._active_zones)}"
            )

        # Expire and update zones
        current_close = c_arr[bar]
        current_high  = h_arr[bar]
        current_low   = l_arr[bar]
        current_open  = o_arr[bar]

        surviving = []
        for z in self._active_zones:
            if abs_bar - z['created_bar'] > ZONE_MAX_BARS:
                continue
            if z['is_bullish'] and current_close < z['fib_bot']:
                continue
            if not z['is_bullish'] and current_close > z['fib_top']:
                continue
            if current_low <= z['fib_top'] and current_high >= z['fib_bot']:
                z['entered_zone'] = True
            surviving.append(z)
        self._active_zones = deque(surviving, maxlen=20)

        # OTE bounce confirmation
        for z in self._active_zones:
            if z.get('signal_fired') or not z['entered_zone'] or abs_bar <= z['created_bar']:
                continue
            if not (current_low <= z['fib_top'] and current_high >= z['fib_bot']):
                continue
            confirmed = (current_close > current_open if z['is_bullish']
                         else current_close < current_open)
            if confirmed:
                z['signal_fired'] = True
                z['entry_bar']    = abs_bar
                z['entry_price']  = current_close
                break

    # ── CISD Feature Vector (10 features) ────────────────────────────────────

    def _build_cisd_feature_vector(
        self, df: pd.DataFrame, abs_bar: int
    ) -> Optional[np.ndarray]:
        """
        Build the 10-element CISD feature vector for the current bar.

        Feature order matches CISD_FEATURE_COLS from training (cisd_ote.py):
          0: zone_height_vs_atr     — zone geometry
          1: price_vs_zone_top      — geometry
          2: price_vs_zone_bot      — geometry
          3: zone_age_bars          — how old the setup is
          4: zone_is_bullish        — direction: +1=BUY, -1=SELL
          5: cisd_displacement_str  — quality of displacement candle
          6: had_liquidity_sweep    — sweep before CISD
          7: entry_distance_pct     — depth into OTE zone
          8: risk_dollars_norm      — stop-loss size normalised
          9: in_optimal_session     — 09:00–11:00 ET flag
        """
        if not self._active_zones:
            self._latest_zone_bullish = 0.0
            return None

        c = float(df['close'].iloc[-1])

        if 'vty_atr_raw' in df.columns:
            atr_raw = float(df['vty_atr_raw'].iloc[-1])
        else:
            h = float(df['high'].iloc[-1]); l_val = float(df['low'].iloc[-1])
            atr_raw = (h - l_val) * 14
        atr_safe = max(atr_raw, 1e-6)

        nearest = None; nearest_dist = float('inf')
        for z in self._active_zones:
            if z.get('signal_fired') and z.get('entry_bar', abs_bar) < abs_bar:
                continue
            mid = (z['fib_top'] + z['fib_bot']) / 2.0
            d   = abs(c - mid)
            if d < nearest_dist:
                nearest_dist = d; nearest = z

        if nearest is None:
            self._latest_zone_bullish = 0.0
            return None

        z = nearest
        self._latest_zone_bullish = 1.0 if z['is_bullish'] else -1.0

        ft = z['fib_top']; fb = z['fib_bot']
        zh = ft - fb; zh_safe = max(zh, 1e-6)
        age = float(abs_bar - z['created_bar'])

        entry_dist = (c - ft) / zh_safe if z['is_bullish'] else (fb - c) / zh_safe

        sl       = fb if z['is_bullish'] else ft
        risk_pts = abs(c - sl)
        pv       = POINT_VALUES.get(self._instrument, 20.0)
        risk_norm = float(np.clip((risk_pts * pv) / MAX_RISK_DOLLARS, 0, 5))

        in_optimal = 0.0
        if hasattr(df.index, 'hour'):
            h_val = int(df.index[-1].hour)
            in_optimal = 1.0 if (OPTIMAL_START_HOUR <= h_val < OPTIMAL_END_HOUR) else 0.0

        features = np.array([
            np.clip(zh / atr_safe, 0, 10),          # 0: zone_height_vs_atr
            np.clip((c - ft) / zh_safe, -2, 5),      # 1: price_vs_zone_top
            np.clip((c - fb) / zh_safe, -2, 5),      # 2: price_vs_zone_bot
            np.clip(age / ZONE_MAX_BARS, 0, 5),       # 3: zone_age_bars
            self._latest_zone_bullish,                # 4: zone_is_bullish
            np.clip(z['disp_strength'], 0, 5),        # 5: cisd_displacement_str
            1.0 if z['had_sweep'] else 0.0,           # 6: had_liquidity_sweep
            float(np.clip(entry_dist, -2, 5)),        # 7: entry_distance_pct
            risk_norm,                                # 8: risk_dollars_norm
            in_optimal,                               # 9: in_optimal_session
        ], dtype=np.float32)

        features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
        return np.clip(features, -10.0, 10.0)
