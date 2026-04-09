"""
Signal Evaluation — 15m-only entry point.

The daily engine has been removed. The ShortTermSignalEngine is now
called directly from main.py. This module is retained for backward
compatibility with any test or import that references it.
"""

from __future__ import annotations

import logging

from coinbase_oracle import CoinbaseOracle
from models import MarketState, RiskParams, Signal
from short_term_engine import ShortTermSignalEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level engine singleton
# ---------------------------------------------------------------------------

_short_term_engine: ShortTermSignalEngine | None = None


def get_short_term_engine() -> ShortTermSignalEngine:
    global _short_term_engine
    if _short_term_engine is None:
        _short_term_engine = ShortTermSignalEngine()
    return _short_term_engine


def evaluate_15m_signal(
    state: MarketState,
    oracle: CoinbaseOracle,
    risk_params: RiskParams,
) -> Signal:
    """Evaluate the 15m signal engine on the current state."""
    engine = get_short_term_engine()
    signal = engine.generate(state, oracle, risk_params)
    logger.info(
        "Signal evaluated | 15m: bias=%s p=%.4f edge=%.4f",
        signal.bias.value, signal.p_prob, signal.edge,
    )
    return signal
