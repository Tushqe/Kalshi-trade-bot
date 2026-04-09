"""
Strike-Conditional Signal Engine for Kalshi 15m BTC Binary Contracts.

Replaces the generic directional engine with per-contract probability
estimation using Geometric Brownian Motion (GBM) hitting probabilities,
microstructure drift overlays, and CF Benchmarks settlement adjustments.

Core formula (all in per-minute units):

    P(S_τ > K) = Φ(d₂)
    d₂ = [ln(S₀/K) + (μ - σ²/2)·τ] / (σ·√τ)

Where:
    S₀  = current BTC mid-price (Coinbase oracle)
    K   = Kalshi contract strike (floor_strike from API)
    τ   = minutes remaining until close_time
    σ   = std dev of 1-minute log returns (realized volatility)
    μ   = microstructure-informed drift per minute (OBI + TFI, capped)
    Φ   = standard normal CDF

Why GBM fits short-dated BTC binaries:
    1. d₂ normalizes distance-to-strike by σ√τ — same dollar gap has
       vastly different probability at 2 min vs 14 min remaining.
    2. Normal CDF provides smooth probability decay across the strike.
    3. Microstructure drift nudges probability without overriding the
       hard math of distance, time, and volatility.
    4. CF Benchmarks BRTI 60s averaging is modeled as vol dampening
       within the last 2 minutes before settlement.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

from coinbase_oracle import CoinbaseOracle
from models import Bias, MarketState, RiskParams, Signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fast normal CDF via math.erf (C-level, <100ns)
# ---------------------------------------------------------------------------

_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF. Uses math.erf for C-level speed."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


# ---------------------------------------------------------------------------
# Strike parser
# ---------------------------------------------------------------------------

_DOLLAR_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")


def parse_strike(market: dict[str, Any]) -> float | None:
    """
    Extract the binary contract strike from a Kalshi market dict.

    Priority order:
    1. ``floor_strike`` / ``cap_strike`` API fields (float, always present
       in real KXBTC15M data — e.g. 68096.44)
    2. Dollar amounts in title/subtitle > $1,000 (BTC-plausible)

    Returns None if unparseable.
    """
    for fld in ("floor_strike", "cap_strike", "strike_price"):
        val = market.get(fld)
        if val is not None:
            try:
                v = float(val)
                if v > 0:
                    return v
            except (ValueError, TypeError):
                pass

    for text_field in ("title", "subtitle", "yes_sub_title"):
        text = market.get(text_field, "")
        for m in _DOLLAR_RE.finditer(text):
            try:
                v = float(m.group(1).replace(",", ""))
                if v > 1000:
                    return v
            except ValueError:
                continue

    return None


# ---------------------------------------------------------------------------
# Realized Volatility Estimator
# ---------------------------------------------------------------------------

def estimate_realized_vol(prices: list[float], window: int = 30) -> float:
    """
    Realized σ of 1-minute log returns over the last ``window`` bars.

    Returns 0.0 if insufficient data (< 5 returns).  Uses sample std dev
    (ddof=1) for unbiased estimation.
    """
    n = min(window + 1, len(prices))
    if n < 6:
        return 0.0
    arr = np.asarray(prices[-n:], dtype=np.float64)
    log_ret = np.diff(np.log(arr))
    return float(np.std(log_ret, ddof=1))


# ---------------------------------------------------------------------------
# Strike-Conditional Engine
# ---------------------------------------------------------------------------

class ShortTermSignalEngine:
    """
    Per-contract probability engine for Kalshi 15m BTC binaries.

    Public API
    ----------
    evaluate_contract(current_price, strike, minutes_remaining, state, oracle)
        → dict with p_above, p_below, sigma, d2, micro_drift, ...

    generate(state, oracle, risk_params)
        → backward-compatible generic Signal for cycle-level logging.
    """

    # Max microstructure probability nudge (applied AFTER base GBM, not
    # inside d₂).  Capped at ±3 percentage points to prevent OBI/TFI from
    # overriding the fundamental distance/time/vol math.
    # Previous bug: drift was inside d₂, accumulating as μ·τ/(σ√τ) = μ√τ/σ,
    # which inflated near-ATM probabilities by 20-54 ppts and caused losses.
    MAX_MICRO_NUDGE = 0.03

    # Internal drift bound used by _compute_micro_drift to scale raw OBI/TFI
    # into a bounded signal.  No longer fed into d₂.
    MAX_MICRO_DRIFT_PER_MIN = 0.0005

    # Volatility floor: prevents near-zero vol from producing
    # unrealistic certainty.  0.0003/min ≈ $30/min on $100k BTC.
    VOL_FLOOR = 0.0003

    # Fallback when we have no price data at all.
    # 0.001/min ≈ $100/min on $100k BTC — wide enough to be conservative.
    VOL_FALLBACK = 0.001

    def __init__(
        self,
        vol_window: int = 30,
        obi_depths: list[int] | None = None,
        tfi_windows: list[float] | None = None,
        micro_weight_obi: float = 0.4,
        micro_weight_tfi: float = 0.6,
        ema_alpha: float = 0.7,
    ) -> None:
        self.vol_window = vol_window
        self.obi_depths = obi_depths or [5, 10, 20]
        self.tfi_windows = tfi_windows or [30.0, 60.0, 300.0]
        self.micro_weight_obi = micro_weight_obi
        self.micro_weight_tfi = micro_weight_tfi

        # EMA smoothing for microstructure drift (prevents oscillation)
        self._ema_alpha = ema_alpha
        self._ema_drift: float | None = None

    # ------------------------------------------------------------------
    # Per-contract evaluation (the core method)
    # ------------------------------------------------------------------

    def evaluate_contract(
        self,
        current_price: float,
        strike: float,
        minutes_remaining: float,
        state: MarketState,
        oracle: CoinbaseOracle,
    ) -> dict[str, float]:
        """
        Compute P(BTC > strike at expiry | S₀, τ, σ, microstructure).

        Parameters
        ----------
        current_price : Current BTC spot price (Binance mid or last).
        strike : Contract strike price in USD (from floor_strike).
        minutes_remaining : Minutes until contract close_time.
        state : Current MarketState (for price history / vol estimation).
        oracle : CoinbaseOracle (for real-time OBI/TFI).

        Returns
        -------
        dict with keys:
            p_above : P(BTC > strike at expiry)
            p_below : 1 - p_above
            sigma : Raw realized vol (per-minute)
            sigma_adj : Vol after settlement damping
            d2 : The d₂ value (for diagnostics)
            micro_drift : Applied drift (per-minute)
            minutes_remaining : Echo back for logging
        """
        # 1. Realized vol from 1-minute bars
        sigma = estimate_realized_vol(state.prices, window=self.vol_window)
        if sigma < self.VOL_FLOOR:
            if sigma == 0.0 and len(state.prices) < 6:
                sigma = self.VOL_FALLBACK
            else:
                sigma = self.VOL_FLOOR

        # 2. CF Benchmarks settlement adjustment
        #    BRTI settles on 60s volume-weighted average.  Within the last
        #    ~2 minutes, the averaging effect reduces effective volatility.
        #    At τ=0: damping → √(1/3) ≈ 0.577.  At τ≥2: no damping.
        sigma_adj = sigma
        if minutes_remaining < 2.0:
            damping = 0.58 + 0.42 * max(0.0, minutes_remaining / 2.0)
            sigma_adj = sigma * damping

        # 3. Microstructure signal (OBI + TFI → bounded probability nudge)
        micro_drift = self._compute_micro_drift(oracle)

        # 4. Base GBM: P(S_τ > K) = Φ(d₂)  — NO drift inside d₂.
        #    Drift is applied as a post-hoc probability nudge (step 5)
        #    to prevent microstructure from dominating near-ATM contracts.
        tau = max(minutes_remaining, 0.01)  # prevent div-by-zero
        vol_sqrt_tau = sigma_adj * math.sqrt(tau)

        if vol_sqrt_tau < 1e-10 or current_price <= 0 or strike <= 0:
            # Degenerate: deterministic outcome
            p_base = 1.0 if current_price > strike else 0.0
            d2 = float("nan")
        else:
            log_moneyness = math.log(current_price / strike)
            # Pure GBM: only the σ²/2 drift correction (no microstructure)
            drift_correction = -(sigma_adj**2 / 2.0) * tau
            d2 = (log_moneyness + drift_correction) / vol_sqrt_tau
            p_base = _norm_cdf(d2)

        # 5. Microstructure nudge: apply OUTSIDE d₂ as a small probability
        #    shift.  This ensures OBI/TFI can never override the hard math
        #    of distance, time, and volatility.
        #    Nudge is scaled by uncertainty: larger nudge when closer to 50%
        #    (ATM), smaller when deep ITM/OTM (already decided).
        uncertainty_scale = 4.0 * p_base * (1.0 - p_base)  # peaks at 1.0 when p=0.5
        micro_nudge = micro_drift / self.MAX_MICRO_DRIFT_PER_MIN * self.MAX_MICRO_NUDGE * uncertainty_scale
        micro_nudge = max(-self.MAX_MICRO_NUDGE, min(self.MAX_MICRO_NUDGE, micro_nudge))
        p_above = p_base + micro_nudge

        # Clamp to [0.01, 0.99] — never express total certainty
        p_above = max(0.01, min(0.99, p_above))

        logger.debug(
            "StrikeEngine | S₀=%.2f K=%.2f τ=%.1fm σ=%.6f σ_adj=%.6f "
            "d₂=%.4f P_base=%.4f nudge=%+.4f → P(above)=%.4f",
            current_price, strike, minutes_remaining,
            sigma, sigma_adj, d2, p_base, micro_nudge, p_above,
        )

        return {
            "p_above": p_above,
            "p_below": 1.0 - p_above,
            "p_base": p_base,
            "sigma": sigma,
            "sigma_adj": sigma_adj,
            "d2": d2,
            "micro_drift": micro_drift,
            "micro_nudge": micro_nudge,
            "minutes_remaining": minutes_remaining,
        }

    # ------------------------------------------------------------------
    # Backward-compatible generic signal (for cycle-level logging)
    # ------------------------------------------------------------------

    def generate(
        self,
        state: MarketState,
        oracle: CoinbaseOracle,
        risk_params: RiskParams,
    ) -> Signal:
        """
        Generic directional signal from microstructure alone.

        Used for cycle-level audit logging; per-contract decisions use
        evaluate_contract() instead.  The p_prob here is NOT used for
        trading when strike data is available.
        """
        obi = self._compute_obi_composite(oracle)
        tfi = self._compute_tfi_composite(oracle)
        micro_drift = self._compute_micro_drift(oracle)
        sigma = estimate_realized_vol(state.prices, window=self.vol_window)

        if micro_drift > 1e-6:
            bias = Bias.BULLISH
            p_prob = min(0.65, 0.5 + abs(micro_drift) * 1000)
        elif micro_drift < -1e-6:
            bias = Bias.BEARISH
            p_prob = max(0.35, 0.5 - abs(micro_drift) * 1000)
        else:
            bias = Bias.NEUTRAL
            p_prob = 0.5

        signal = Signal(
            bias=bias,
            p_prob=p_prob,
            edge=p_prob - state.kalshi_15m_implied,
            horizon="15m",
            confidence=min(1.0, (abs(obi) + abs(tfi)) / 2.0),
            metadata={
                "obi": obi,
                "tfi": tfi,
                "micro_drift": micro_drift,
                "sigma_1min": sigma,
                "engine": "strike_conditional",
            },
        )

        logger.info(
            "StrikeEngine (generic) | bias=%s p=%.4f σ=%.6f μ=%.7f "
            "obi=%.3f tfi=%.3f",
            bias.value, p_prob, sigma, micro_drift, obi, tfi,
        )
        return signal

    # ------------------------------------------------------------------
    # Internal: microstructure helpers
    # ------------------------------------------------------------------

    def _compute_obi_composite(self, oracle: CoinbaseOracle) -> float:
        """Multi-depth weighted OBI.  Deeper levels get less weight."""
        scores = [oracle.compute_obi(depth_levels=d) for d in self.obi_depths]
        weights = [1.0 / (i + 1) for i in range(len(scores))]
        w_sum = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / w_sum

    def _compute_tfi_composite(self, oracle: CoinbaseOracle) -> float:
        """Multi-window weighted TFI.  Shorter windows get more weight."""
        scores: list[float] = []
        for window in self.tfi_windows:
            now = time.monotonic()
            cutoff = now - window
            net = 0.0
            total = 0.0
            for ts, sq in oracle._trades:
                if ts >= cutoff:
                    net += sq
                    total += abs(sq)
            scores.append(net / total if total > 0 else 0.0)

        weights = [1.0 / (i + 1) for i in range(len(scores))]
        w_sum = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / w_sum

    def _compute_micro_drift(self, oracle: CoinbaseOracle) -> float:
        """
        Convert OBI + TFI into a per-minute drift estimate.

        The raw signal (in [-1, 1]) is scaled to a bounded drift, then
        EMA-smoothed to prevent oscillation between 10-second cycles.
        """
        obi = self._compute_obi_composite(oracle)
        tfi = self._compute_tfi_composite(oracle)

        raw_signal = (
            self.micro_weight_obi * obi
            + self.micro_weight_tfi * tfi
        )

        # Map [-1, 1] signal to bounded drift per minute
        raw_drift = raw_signal * self.MAX_MICRO_DRIFT_PER_MIN

        # EMA smoothing
        if self._ema_drift is None:
            self._ema_drift = raw_drift
        else:
            self._ema_drift = (
                self._ema_alpha * raw_drift
                + (1.0 - self._ema_alpha) * self._ema_drift
            )

        return self._ema_drift
