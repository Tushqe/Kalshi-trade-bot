"""
Trading Strategies.

Four concrete strategies inheriting from ``BaseStrategy``, each producing a
``StrategySignal`` with a directional score in [-1, 1] and a confidence in
[0, 1].  The ensemble engine consumes these signals.

All calculations operate on the ``MarketState.prices`` array where the
*last* element is the most recent observation.
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np

from models import BaseStrategy, MarketState, Regime, StrategySignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(values: Sequence[float], span: int) -> np.ndarray:
    """Exponential moving average using the standard decay factor."""
    arr = np.asarray(values, dtype=np.float64)
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _sma(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    cumsum = np.cumsum(arr)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    out = np.full_like(arr, np.nan)
    out[window - 1:] = cumsum[window - 1:] / window
    return out


def _std_rolling(values: Sequence[float], window: int) -> np.ndarray:
    """Rolling standard deviation (population) using a simple loop."""
    arr = np.asarray(values, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    for i in range(window - 1, len(arr)):
        out[i] = np.std(arr[i - window + 1: i + 1])
    return out


def _rsi(prices: Sequence[float], period: int = 14) -> float:
    """
    Relative Strength Index of the most recent bar.

    RSI = 100 - 100 / (1 + avg_gain / avg_loss)
    Uses Wilder's smoothing (EMA with alpha = 1/period).
    """
    arr = np.asarray(prices, dtype=np.float64)
    if len(arr) < period + 1:
        return 50.0  # neutral when insufficient data

    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    alpha = 1.0 / period
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    for i in range(period, len(gains)):
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


# ---------------------------------------------------------------------------
# 1. Trend Following (Time-Series Momentum)
# ---------------------------------------------------------------------------

class TrendFollowingStrategy(BaseStrategy):
    """
    Dual EMA cross-over momentum with rate-of-change confirmation.

    Signal logic
    ------------
    - Compute fast EMA (12) and slow EMA (26) of close prices.
    - MACD = fast - slow.  Signal line = EMA(9) of MACD.
    - Normalise MACD histogram to [-1, 1] using the ATR-scaled denominator.
    - Confidence = min(1, |histogram| / atr) to reflect signal clarity.
    """

    name = "trend_following"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    async def generate_signal(self, state: MarketState) -> StrategySignal:
        prices = state.prices
        min_len = self.slow + self.signal_period
        if len(prices) < min_len:
            return StrategySignal(strategy_name=self.name, score=0.0, confidence=0.0,
                                  metadata={"reason": "insufficient_data"})

        fast_ema = _ema(prices, self.fast)
        slow_ema = _ema(prices, self.slow)
        macd_line = fast_ema - slow_ema
        signal_line = _ema(macd_line.tolist(), self.signal_period)
        histogram = macd_line - signal_line

        # ATR proxy: mean absolute price change over slow window
        abs_changes = np.abs(np.diff(prices[-self.slow:]))
        atr = float(abs_changes.mean()) if len(abs_changes) > 0 else 1.0
        atr = max(atr, 1e-8)

        raw_score = float(histogram[-1]) / atr
        score = max(-1.0, min(1.0, raw_score))
        confidence = min(1.0, abs(raw_score))

        logger.debug("TrendFollowing | macd_hist=%.2f atr=%.2f score=%.3f conf=%.3f",
                      histogram[-1], atr, score, confidence)
        return StrategySignal(strategy_name=self.name, score=score, confidence=confidence,
                              metadata={"macd_hist": float(histogram[-1]), "atr": atr})


# ---------------------------------------------------------------------------
# 2. Mean Reversion (Bollinger Band + RSI)
# ---------------------------------------------------------------------------

class MeanReversionStrategy(BaseStrategy):
    """
    Intraday reversal strategy using Bollinger Bands and RSI.

    Signal logic
    ------------
    - Compute 20-period SMA and 2σ Bollinger Bands.
    - z_score = (price - SMA) / σ  → normalised distance from mean.
    - RSI(14): oversold < 30, overbought > 70.
    - Score = -z_score (mean-reverting: price above band → bearish).
    - Confidence boosted when RSI confirms the z-score direction.
    """

    name = "mean_reversion"

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0, rsi_period: int = 14) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period

    async def generate_signal(self, state: MarketState) -> StrategySignal:
        prices = state.prices
        if len(prices) < self.bb_period:
            return StrategySignal(strategy_name=self.name, score=0.0, confidence=0.0,
                                  metadata={"reason": "insufficient_data"})

        sma = _sma(prices, self.bb_period)
        std = _std_rolling(prices, self.bb_period)

        current_sma = float(sma[-1])
        current_std = float(std[-1])
        if math.isnan(current_sma) or current_std < 1e-8:
            return StrategySignal(strategy_name=self.name, score=0.0, confidence=0.0)

        z = (prices[-1] - current_sma) / current_std
        rsi_val = _rsi(prices, self.rsi_period)

        # Mean-reverting: flip the z-score
        raw_score = -z / self.bb_std  # normalise so ±2σ → ±1
        score = max(-1.0, min(1.0, raw_score))

        # Confidence: base from |z|, boosted if RSI confirms
        base_conf = min(1.0, abs(z) / self.bb_std)
        rsi_confirms = (z > 0 and rsi_val > 70) or (z < 0 and rsi_val < 30)
        confidence = min(1.0, base_conf * (1.3 if rsi_confirms else 0.8))

        logger.debug("MeanReversion | z=%.2f rsi=%.1f score=%.3f conf=%.3f",
                      z, rsi_val, score, confidence)
        return StrategySignal(strategy_name=self.name, score=score, confidence=confidence,
                              metadata={"z_score": z, "rsi": rsi_val})


# ---------------------------------------------------------------------------
# 3. Microstructure Strategy
# ---------------------------------------------------------------------------

class MicrostructureStrategy(BaseStrategy):
    """
    Short-term direction from order book and trade flow imbalance.

    Directly consumes OBI and TFI from the ``MarketState`` (populated by
    the Binance Oracle) and emits a weighted signal.
    """

    name = "microstructure"

    def __init__(self, obi_weight: float = 0.4, tfi_weight: float = 0.6) -> None:
        self.obi_w = obi_weight
        self.tfi_w = tfi_weight

    async def generate_signal(self, state: MarketState) -> StrategySignal:
        obi = state.order_book_imbalance
        tfi = state.trade_flow_imbalance

        raw = self.obi_w * obi + self.tfi_w * tfi
        score = max(-1.0, min(1.0, raw))

        # Confidence rises when OBI and TFI agree in direction
        agreement = 1.0 if (obi * tfi > 0) else 0.5
        confidence = min(1.0, abs(raw) * agreement)

        logger.debug("Microstructure | obi=%.3f tfi=%.3f score=%.3f conf=%.3f",
                      obi, tfi, score, confidence)
        return StrategySignal(strategy_name=self.name, score=score, confidence=confidence,
                              metadata={"obi": obi, "tfi": tfi})


# ---------------------------------------------------------------------------
# 4. Regime / Volatility Strategy (Mock ML Inference)
# ---------------------------------------------------------------------------

class RegimeVolatilityStrategy(BaseStrategy):
    """
    Daily regime classification producing a directional bias.

    In production this would call an ML model (e.g., HMM, gradient-boosted
    classifier).  Here we implement a rule-based proxy:

    1. Realised volatility = std(log-returns) over a lookback.
    2. If vol is in the top quartile → ``HIGH_VOLATILITY`` regime.
    3. Momentum direction over the lookback sets the sign.
    4. Trending + moderate vol → strong signal; high vol → low confidence.
    """

    name = "regime_volatility"

    def __init__(self, lookback: int = 30, vol_threshold_high: float = 0.75) -> None:
        self.lookback = lookback
        self.vol_q_high = vol_threshold_high

    async def generate_signal(self, state: MarketState) -> StrategySignal:
        prices = state.prices
        if len(prices) < self.lookback + 1:
            return StrategySignal(strategy_name=self.name, score=0.0, confidence=0.0,
                                  metadata={"reason": "insufficient_data"})

        arr = np.asarray(prices[-self.lookback - 1:], dtype=np.float64)
        log_returns = np.diff(np.log(arr))

        realised_vol = float(np.std(log_returns))
        momentum = float(np.sum(log_returns))  # cumulative log return

        # Classify regime
        # Use historical vol percentile (simplified: compare to a fixed threshold)
        vol_percentile = min(1.0, realised_vol / 0.03)  # 3 % daily vol ≈ ceiling
        if vol_percentile > self.vol_q_high:
            regime = Regime.HIGH_VOLATILITY
        elif vol_percentile < 0.25:
            regime = Regime.LOW_VOLATILITY
        elif abs(momentum) > realised_vol:
            regime = Regime.TRENDING
        else:
            regime = Regime.MEAN_REVERTING

        # Score: direction from momentum, scaled by conviction
        if regime == Regime.TRENDING:
            score = max(-1.0, min(1.0, momentum / (realised_vol + 1e-8)))
            confidence = min(1.0, abs(momentum) / (2 * realised_vol + 1e-8))
        elif regime == Regime.HIGH_VOLATILITY:
            # Cautious: still directional but low confidence
            score = max(-1.0, min(1.0, momentum / (realised_vol + 1e-8) * 0.5))
            confidence = 0.3
        elif regime == Regime.MEAN_REVERTING:
            # Counter-trend lean
            score = max(-1.0, min(1.0, -momentum / (realised_vol + 1e-8) * 0.3))
            confidence = 0.4
        else:
            score = 0.0
            confidence = 0.2

        logger.debug("RegimeVol | vol=%.4f mom=%.4f regime=%s score=%.3f conf=%.3f",
                      realised_vol, momentum, regime.value, score, confidence)
        return StrategySignal(
            strategy_name=self.name, score=score, confidence=confidence,
            metadata={"realised_vol": realised_vol, "momentum": momentum, "regime": regime.value},
        )
