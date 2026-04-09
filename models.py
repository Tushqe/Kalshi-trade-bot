"""
Core Interfaces, Data Models & Abstract Strategy Base.

Defines the canonical data types flowing through the trading pipeline:
MarketState -> StrategySignal -> EnsemblePrediction -> TradeInstruction.

Extended with multi-horizon Signal, RiskParams, and ChosenContract for
the dual daily/15m trading architecture.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Side(str, Enum):
    YES = "yes"
    NO = "no"


class OrderAction(str, Enum):
    BUY = "buy"
    SELL = "sell"


class Regime(str, Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


class Bias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class LiquidityTier(str, Enum):
    DEEP = "deep"                      # Multiple levels, tight spread, ≥50 contracts
    THIN_TRADEABLE = "thin_tradeable"  # 2+ levels, ≥10 contracts, ≤10c spread
    THIN_MARGINAL = "thin_marginal"    # Some WS depth but fails THIN_TRADEABLE
    EMPTY = "empty"                    # Zero WS levels but valid listing prices
    DEAD = "dead"                      # No listing prices, no data at all

    # Backwards compat: old code may compare against "thin"
    @classmethod
    def _missing_(cls, value: object) -> "LiquidityTier | None":
        if value == "thin":
            return cls.THIN_TRADEABLE
        return None

    @property
    def is_thin(self) -> bool:
        """True for either THIN sub-tier (replaces old == THIN checks)."""
        return self in (LiquidityTier.THIN_TRADEABLE, LiquidityTier.THIN_MARGINAL)


class ExecutionMode(str, Enum):
    PASSIVE = "passive"       # Post inside spread, wait for fill
    MODERATE = "moderate"     # Post near mid, shorter patience (~1 min)
    AGGRESSIVE = "aggressive" # Take the best ask immediately
    SKIP = "skip"             # Don't trade this opportunity


# ---------------------------------------------------------------------------
# Market State — snapshot consumed by every strategy
# ---------------------------------------------------------------------------

class MarketState(BaseModel):
    """Unified snapshot of the current market environment."""

    symbol: str = "BTCUSDT"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Price series (most-recent last)
    prices: list[float] = Field(default_factory=list, description="Recent close prices (1m)")
    volumes: list[float] = Field(default_factory=list, description="Recent volumes (1m)")
    highs: list[float] = Field(default_factory=list, description="Recent highs (1m)")
    lows: list[float] = Field(default_factory=list, description="Recent lows (1m)")

    # Multi-timeframe OHLCV for daily engine
    prices_1h: list[float] = Field(default_factory=list, description="1h close prices")
    volumes_1h: list[float] = Field(default_factory=list, description="1h volumes")
    highs_1h: list[float] = Field(default_factory=list, description="1h highs")
    lows_1h: list[float] = Field(default_factory=list, description="1h lows")

    prices_4h: list[float] = Field(default_factory=list, description="4h close prices")
    prices_1d: list[float] = Field(default_factory=list, description="1d close prices")

    # Microstructure fields populated by Binance oracle
    order_book_imbalance: float = 0.0
    trade_flow_imbalance: float = 0.0
    mid_price: float = 0.0
    spread_bps: float = 0.0

    # Short-term derived
    return_15m: float = 0.0       # 15-minute return (latest)
    volatility_1h: float = 0.0    # 1h realised volatility (log-return std)

    # Kalshi implied odds (populated from order books)
    kalshi_daily_implied: float = 0.5
    kalshi_15m_implied: float = 0.5

    # Derived / external
    regime: Regime = Regime.UNKNOWN

    @field_validator("prices", "volumes", mode="before")
    @classmethod
    def _coerce_floats(cls, v: list) -> list[float]:
        return [float(x) for x in v]


# ---------------------------------------------------------------------------
# Multi-Horizon Signal
# ---------------------------------------------------------------------------

class Signal(BaseModel):
    """Output of a horizon-specific signal engine."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    bias: Bias = Bias.NEUTRAL
    p_prob: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Model probability that BTC goes up in the given horizon",
    )
    edge: float = Field(
        default=0.0,
        description="p_prob minus Kalshi implied ask probability",
    )
    horizon: Literal["daily", "15m"] = "daily"
    confidence: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Overall signal confidence (blend of sub-model confidences)",
    )
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Risk Parameters (injectable configuration)
# ---------------------------------------------------------------------------

class RiskParams(BaseModel):
    """Configurable risk and routing parameters for the pure 15m bot."""

    # Position sizing
    max_risk_pct: float = Field(default=0.05, description="Max % of equity per trade")
    max_daily_loss_pct: float = Field(default=0.10, description="Daily drawdown circuit breaker")
    kelly_fraction: float = Field(default=0.25, description="Kelly scale-down factor")

    # 15m caps
    max_risk_pct_15m: float = Field(default=0.03, description="Max risk per 15m trade")
    max_open_positions: int = Field(default=3)

    # 15m router thresholds
    min_15m_edge: float = Field(default=0.04, description="Min edge to trade 15m contracts")
    max_15m_spread_pts: float = Field(default=0.15, description="Max bid-ask spread for 15m")
    min_15m_oi: int = Field(default=0, description="Min open interest for 15m")
    max_volatility_for_15m: float = Field(
        default=0.02,
        description="If 1h realised vol exceeds this, skip cycle",
    )


# ---------------------------------------------------------------------------
# Chosen Contract — output of the ContractRouter
# ---------------------------------------------------------------------------

class ChosenContract(BaseModel):
    """The contract selected by the router for execution."""

    ticker: str
    horizon: Literal["daily", "15m"]
    side: Side = Side.YES
    p_model: float = Field(ge=0.0, le=1.0)
    kalshi_ask: float = Field(ge=0.0, le=1.0, description="Best ask as probability (0-1)")
    edge: float
    spread: float = Field(default=0.0, description="Bid-ask spread in probability points")
    open_interest: int = Field(default=0)
    liquidity_tier: LiquidityTier = LiquidityTier.DEEP
    close_time: Optional[datetime] = None
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Strategy Signal — output of a single strategy (legacy, still used by ensemble)
# ---------------------------------------------------------------------------

class StrategySignal(BaseModel):
    """Output of one strategy evaluation cycle."""

    strategy_name: str
    score: float = Field(
        ge=-1.0, le=1.0,
        description="Directional score: +1 = max bullish, -1 = max bearish",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Self-assessed confidence in the signal",
    )
    metadata: dict = Field(default_factory=dict)

    @property
    def weighted_contribution(self) -> float:
        return self.score * self.confidence


# ---------------------------------------------------------------------------
# Ensemble Prediction — aggregated view before oracle adjustment
# ---------------------------------------------------------------------------

class EnsemblePrediction(BaseModel):
    """Combined output of the ensemble engine."""

    base_probability: float = Field(
        ge=0.0, le=1.0,
        description="Pre-oracle probability of the YES outcome",
    )
    raw_score: float = Field(description="Unbounded weighted consensus score")
    bias: float = Field(
        default=0.0,
        description="Oracle-applied microstructure bias in probability space",
    )
    component_signals: list[StrategySignal] = Field(default_factory=list)

    @property
    def adjusted_probability(self) -> float:
        """Probability after oracle bias, clamped to [0.01, 0.99]."""
        return max(0.01, min(0.99, self.base_probability + self.bias))


# ---------------------------------------------------------------------------
# Trade Instruction — final order payload
# ---------------------------------------------------------------------------

class TradeInstruction(BaseModel):
    """Fully-qualified order ready for submission to Kalshi."""

    ticker: str
    action: OrderAction
    side: Side
    contracts: int = Field(ge=1)
    limit_price_cents: int = Field(
        ge=1, le=99,
        description="Limit price in cents (Kalshi YES contracts trade 1-99c)",
    )
    model_probability: float
    edge: float = Field(description="model_prob - implied_prob of the order")
    horizon: Literal["daily", "15m"] = "daily"
    liquidity_tier: LiquidityTier = LiquidityTier.DEEP
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pending Order — tracks resting limit orders awaiting fill
# ---------------------------------------------------------------------------

class PendingOrder(BaseModel):
    """A resting limit order submitted to Kalshi, tracked until resolved."""

    order_id: str
    ticker: str
    side: Side
    action: OrderAction = OrderAction.BUY
    limit_price_cents: int = Field(ge=1, le=99)
    contracts: int = Field(ge=1)
    horizon: Literal["daily", "15m"] = "daily"
    liquidity_tier: LiquidityTier = LiquidityTier.EMPTY
    model_p_at_placement: float
    edge_at_placement: float
    placed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expiry_utc: Optional[datetime] = None
    cycles_alive: int = 0
    status: Literal["resting", "filled", "cancelled", "expired"] = "resting"
    execution_mode: str = "stink"    # passive, moderate, stink, aggressive
    strike: Optional[float] = None   # for model re-evaluation during reevaluation
    metadata: dict = Field(default_factory=dict)  # strategy-specific data (reference_bid, escalation state)


# ---------------------------------------------------------------------------
# Deferred Watchlist — tracks skipped opportunities for retry
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

@dataclass
class WatchlistEntry:
    """A contract that was skipped due to liquidity but had edge."""
    ticker: str
    side: Side
    model_p: float
    edge_at_discovery: float
    close_time: datetime
    skip_reason: str
    strike: float | None = None
    retry_count: int = 0
    max_retries: int = 5
    market_snapshot: dict = field(default_factory=dict)


class DeferredWatchlist:
    """
    Tracks DEAD/EMPTY/THIN_MARGINAL contracts that had edge >= min_edge.
    Re-evaluates each cycle for up to max_retries cycles (50 seconds).
    Entries are purged when close_time approaches or retries exhausted.
    """

    def __init__(self, max_entries: int = 10):
        self._entries: dict[str, WatchlistEntry] = {}
        self.max_entries = max_entries

    def add(self, entry: WatchlistEntry) -> None:
        secs_left = (entry.close_time - datetime.now(timezone.utc)).total_seconds()
        if secs_left < 60:
            return  # Not enough time remaining to be worth watching

        if entry.ticker in self._entries:
            existing = self._entries[entry.ticker]
            if entry.edge_at_discovery > existing.edge_at_discovery:
                self._entries[entry.ticker] = entry
            return

        if len(self._entries) >= self.max_entries:
            worst_key = min(
                self._entries,
                key=lambda k: self._entries[k].edge_at_discovery,
            )
            if entry.edge_at_discovery > self._entries[worst_key].edge_at_discovery:
                del self._entries[worst_key]
            else:
                return

        self._entries[entry.ticker] = entry

    def get_retry_candidates(self) -> list[WatchlistEntry]:
        """Return entries worth re-evaluating, purging expired ones."""
        now = datetime.now(timezone.utc)
        expired = [
            k for k, v in self._entries.items()
            if (v.close_time - now).total_seconds() < 30
            or v.retry_count >= v.max_retries
        ]
        for k in expired:
            del self._entries[k]

        return list(self._entries.values())

    def mark_retried(self, ticker: str) -> None:
        if ticker in self._entries:
            self._entries[ticker].retry_count += 1

    def remove(self, ticker: str) -> None:
        self._entries.pop(ticker, None)

    def purge_expired_tickers(self, active_tickers: set[str]) -> None:
        to_remove = [t for t in self._entries if t not in active_tickers]
        for t in to_remove:
            del self._entries[t]

    @property
    def size(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# Abstract Strategy Base
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """
    Contract every strategy must fulfil.

    Subclasses implement ``generate_signal`` which receives the current
    ``MarketState`` and returns a ``StrategySignal``.
    """

    name: str = "base"

    @abstractmethod
    async def generate_signal(self, state: MarketState) -> StrategySignal:
        """Produce a directional signal from the current market snapshot."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
