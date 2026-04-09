"""
Risk Management Engine.

Enforces per-trade sizing via fractional Kelly, daily drawdown circuit
breakers, and maximum allocation caps.  Every trade instruction passes
through ``RiskManager`` before reaching the exchange.

Extended with ``PositionSizer`` for dual-horizon (daily / 15m) sizing
with horizon-aware caps.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Literal

from models import ChosenContract, LiquidityTier, RiskParams, TradeInstruction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration (legacy — still used by RiskManager)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskConfig:
    """Immutable risk parameters — set once at startup."""

    # Account / sizing
    initial_equity_cents: int = 10_000_00          # $10,000 in cents
    max_allocation_pct: float = 0.02               # 2 % of equity per trade
    kelly_fraction: float = 0.25                   # Quarter-Kelly (conservative)

    # Drawdown
    max_daily_drawdown_pct: float = 0.05           # 5 % daily loss → circuit breaker
    max_open_positions: int = 3                    # aligned with RiskParams

    # Edge filter
    min_edge_threshold: float = 0.015              # 1.5 % minimum expected edge


# ---------------------------------------------------------------------------
# Daily P&L Tracker
# ---------------------------------------------------------------------------

@dataclass
class DailyPnL:
    """Running P&L accumulator for the current trading day."""

    date_key: date = field(default_factory=lambda: datetime.now(timezone.utc).date())
    realised_pnl_cents: int = 0
    peak_equity_cents: int = 0
    trade_count: int = 0

    def reset_if_new_day(self, equity_cents: int) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self.date_key:
            logger.info(
                "New trading day %s — resetting daily P&L (prev day P&L: %d cents)",
                today, self.realised_pnl_cents,
            )
            self.date_key = today
            self.realised_pnl_cents = 0
            self.peak_equity_cents = equity_cents
            self.trade_count = 0

    def record_fill(self, pnl_cents: int) -> None:
        self.realised_pnl_cents += pnl_cents
        self.trade_count += 1


# ---------------------------------------------------------------------------
# Risk Manager (core gate — shared by both old and new paths)
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Central risk gate.  Call ``passes_risk_checks`` before sending any order.

    Responsibilities
    ----------------
    1. **Kelly sizing** — compute optimal position size from model edge.
    2. **Allocation cap** — never risk more than ``max_allocation_pct`` of equity.
    3. **Drawdown breaker** — halt trading when daily losses exceed threshold.
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.cfg = config or RiskConfig()
        self.equity_cents: int = self.cfg.initial_equity_cents
        self.daily = DailyPnL(peak_equity_cents=self.equity_cents)
        self.open_position_count: int = 0
        # Dual-track risk: resting orders tracked separately from filled positions
        self.resting_order_count: int = 0
        self.resting_risk_cents: int = 0
        logger.info("RiskManager initialised | equity=%d | config=%s", self.equity_cents, self.cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync_equity(self, balance_cents: int) -> None:
        """Update equity from an external source (e.g. Kalshi account balance).

        On first sync (when equity still equals the configured initial value),
        reset peak_equity to the real balance so the drawdown breaker uses a
        realistic baseline instead of the placeholder initial config value.
        """
        old = self.equity_cents
        first_sync = (old == self.cfg.initial_equity_cents and balance_cents != old)
        self.equity_cents = balance_cents
        if first_sync:
            # Bootstrap: adopt the real balance as peak so drawdown calc is sane
            self.daily.peak_equity_cents = balance_cents
            self.daily.realised_pnl_cents = 0
            logger.info(
                "First equity sync — resetting peak to real balance: %d cents ($%.2f)",
                balance_cents, balance_cents / 100,
            )
        elif balance_cents > self.daily.peak_equity_cents:
            self.daily.peak_equity_cents = balance_cents
        logger.info("Equity synced: %d -> %d cents ($%.2f)",
                    old, balance_cents, balance_cents / 100)

    def passes_risk_checks(self, edge: float) -> tuple[bool, str]:
        """
        Return (allowed, reason).  ``reason`` is empty on success.
        """
        self.daily.reset_if_new_day(self.equity_cents)

        # 1. Minimum edge
        if edge < self.cfg.min_edge_threshold:
            return False, f"Edge {edge:.4f} below threshold {self.cfg.min_edge_threshold}"

        # 2. Daily drawdown circuit breaker
        dd_pct = self._daily_drawdown_pct()
        if dd_pct >= self.cfg.max_daily_drawdown_pct:
            return False, (
                f"Daily drawdown {dd_pct:.2%} exceeds limit "
                f"{self.cfg.max_daily_drawdown_pct:.2%} — CIRCUIT BREAKER"
            )

        # 3. Open position cap
        if self.open_position_count >= self.cfg.max_open_positions:
            return False, f"Open positions ({self.open_position_count}) at maximum"

        return True, ""

    def compute_kelly_contracts(
        self,
        model_prob: float,
        ask_price_cents: int,
    ) -> int:
        """
        Fractional Kelly criterion for binary (YES/NO) contracts.

        Full Kelly fraction for a binary bet paying $1:
            f* = (p * b - q) / b
        where
            p = model probability of YES
            q = 1 - p
            b = (100 - ask) / ask   (net odds offered by the market)

        We then scale by ``kelly_fraction`` (quarter-Kelly) and cap at
        ``max_allocation_pct`` of current equity.

        Returns the number of contracts (integer >= 0).
        """
        if ask_price_cents <= 0 or ask_price_cents >= 100:
            logger.warning("Invalid ask price %d — refusing to size", ask_price_cents)
            return 0

        p = model_prob
        q = 1.0 - p
        b = (100 - ask_price_cents) / ask_price_cents  # payout odds

        if b <= 0:
            return 0

        full_kelly = (p * b - q) / b
        if full_kelly <= 0:
            logger.debug("Kelly fraction non-positive (%.4f) — no trade", full_kelly)
            return 0

        fraction = full_kelly * self.cfg.kelly_fraction

        # Dollar amount to risk (in cents)
        risk_cents = fraction * self.equity_cents

        # Cap at max allocation
        max_risk_cents = self.cfg.max_allocation_pct * self.equity_cents
        risk_cents = min(risk_cents, max_risk_cents)

        # Each contract costs ``ask_price_cents``
        contracts = int(math.floor(risk_cents / ask_price_cents))
        contracts = max(contracts, 0)

        logger.info(
            "Kelly sizing | p=%.3f ask=%dc b=%.2f f*=%.4f frac=%.4f "
            "risk=%dc max=%dc -> %d contracts",
            p, ask_price_cents, b, full_kelly, fraction,
            risk_cents, max_risk_cents, contracts,
        )
        return contracts

    def record_trade_result(self, pnl_cents: int) -> None:
        """Update equity and daily tracker after a fill / settlement."""
        self.equity_cents += pnl_cents
        self.daily.record_fill(pnl_cents)
        if self.equity_cents > self.daily.peak_equity_cents:
            self.daily.peak_equity_cents = self.equity_cents
        logger.info(
            "Trade result recorded | pnl=%d equity=%d daily_pnl=%d",
            pnl_cents, self.equity_cents, self.daily.realised_pnl_cents,
        )

    def notify_position_opened(self) -> None:
        self.open_position_count += 1

    def notify_position_closed(self) -> None:
        self.open_position_count = max(0, self.open_position_count - 1)

    def sync_position_count(self, actual_count: int) -> None:
        """Reconcile open_position_count with the real count from the exchange."""
        old = self.open_position_count
        if old != actual_count:
            logger.info(
                "Position count reconciled: internal=%d -> actual=%d",
                old, actual_count,
            )
        self.open_position_count = actual_count

    # ------------------------------------------------------------------
    # Stink Bid Risk Gate (resting limit orders)
    # ------------------------------------------------------------------

    def passes_risk_checks_for_stink_bid(
        self, edge: float, cost_cents: int,
    ) -> tuple[bool, str]:
        """
        Separate risk gate for stink bids (resting limit orders).
        More conservative on drawdown (8% pre-emption vs 10% breaker).
        Separate cap on resting orders (don't count against filled positions).
        """
        self.daily.reset_if_new_day(self.equity_cents)

        # 1. Minimum edge
        if edge < self.cfg.min_edge_threshold:
            return False, f"Edge {edge:.4f} below threshold"

        # 2. Daily drawdown — pre-emptive at 8%
        dd_pct = self._daily_drawdown_pct()
        if dd_pct >= 0.08:
            return False, f"Stink bid blocked — drawdown at {dd_pct:.2%} (approaching breaker)"

        # 3. Resting order cap
        if self.resting_order_count >= 7:
            return False, f"Resting orders ({self.resting_order_count}) at cap"

        # 4. Total risk exposure (filled + resting) must not exceed 25% of equity
        total_risk = self.resting_risk_cents + cost_cents
        if self.equity_cents > 0 and total_risk > self.equity_cents * 0.25:
            return False, (
                f"Total resting risk ({total_risk}c) would exceed "
                f"25% of equity ({self.equity_cents}c)"
            )

        return True, ""

    def notify_resting_order_placed(self, cost_cents: int) -> None:
        """Track a new resting order's risk exposure."""
        self.resting_order_count += 1
        self.resting_risk_cents += cost_cents
        logger.debug(
            "Resting order placed | count=%d risk=%dc",
            self.resting_order_count, self.resting_risk_cents,
        )

    def notify_resting_order_resolved(self, cost_cents: int, was_filled: bool) -> None:
        """A resting order was filled or cancelled."""
        self.resting_order_count = max(0, self.resting_order_count - 1)
        self.resting_risk_cents = max(0, self.resting_risk_cents - cost_cents)
        if was_filled:
            self.notify_position_opened()
        logger.debug(
            "Resting order resolved | filled=%s count=%d risk=%dc",
            was_filled, self.resting_order_count, self.resting_risk_cents,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _daily_drawdown_pct(self) -> float:
        peak = self.daily.peak_equity_cents
        if peak <= 0:
            return 0.0
        return (peak - self.equity_cents) / peak


# ---------------------------------------------------------------------------
# Position Sizer — dual-horizon Kelly sizing
# ---------------------------------------------------------------------------

class PositionSizer:
    """
    Computes the number of contracts to buy for a ``ChosenContract``,
    using Kelly-style fraction with horizon-aware caps.

    Kelly for binary contracts:
        f* = (p_model - kalshi_ask) / (1 - kalshi_ask)

    Then:
        scaled = f* * kelly_fraction  (e.g., quarter-Kelly)
        capped  = min(scaled, max_risk_pct)   for daily
                  min(scaled, max_risk_pct_15m) for 15m

    Contracts = floor(risk_cents / ask_price_cents)
    """

    def __init__(self, risk_params: RiskParams, risk_manager: RiskManager) -> None:
        self.params = risk_params
        self.rm = risk_manager

    def size(self, contract: ChosenContract) -> int:
        """
        Compute number of contracts based on Kelly criterion and horizon caps.

        Parameters
        ----------
        contract : ChosenContract
            The selected contract from the router.

        Returns
        -------
        int
            Number of contracts to buy (>= 0).
        """
        p_model = contract.p_model
        kalshi_ask = contract.kalshi_ask
        horizon = contract.horizon

        # ------------------------------------------------------------------
        # Risk gate: check drawdown / position limits
        # Stink bids use a separate risk gate (don't count against position limit)
        # ------------------------------------------------------------------
        tier = contract.liquidity_tier
        if tier == LiquidityTier.EMPTY:
            # Stink bid risk gate is checked in kalshi_client.execute_stink_bid()
            # Here just check basic edge threshold
            if contract.edge < self.rm.cfg.min_edge_threshold:
                logger.info("PositionSizer | blocked: edge below threshold for stink bid")
                return 0
        else:
            allowed, reason = self.rm.passes_risk_checks(contract.edge)
            if not allowed:
                logger.info("PositionSizer | blocked: %s", reason)
                return 0

        # ------------------------------------------------------------------
        # Kelly fraction: f* = (p_model - kalshi_ask) / (1 - kalshi_ask)
        # ------------------------------------------------------------------
        if kalshi_ask >= 1.0 or kalshi_ask <= 0.0:
            logger.warning("Invalid kalshi_ask=%.4f — refusing to size", kalshi_ask)
            return 0

        full_kelly = (p_model - kalshi_ask) / (1.0 - kalshi_ask)

        if full_kelly <= 0:
            logger.debug("Kelly non-positive (%.4f) — no trade", full_kelly)
            return 0

        # Scale down
        scaled = full_kelly * self.params.kelly_fraction

        # Horizon-specific cap
        if horizon == "15m":
            max_pct = self.params.max_risk_pct_15m
        else:
            max_pct = self.params.max_risk_pct

        capped = min(scaled, max_pct)

        # ------------------------------------------------------------------
        # Convert to contract count
        # ------------------------------------------------------------------
        equity = self.rm.equity_cents
        risk_cents = capped * equity

        # Daily loss cap: ensure this trade + existing daily losses don't exceed limit
        self.rm.daily.reset_if_new_day(equity)
        # Only subtract actual losses from budget; profits should NOT shrink sizing.
        # abs() was treating winning days as losing days — the #1 P&L killer.
        daily_losses = max(0, -self.rm.daily.realised_pnl_cents)
        remaining_daily_budget = (
            self.params.max_daily_loss_pct * equity
            - daily_losses
        )
        if remaining_daily_budget <= 0:
            logger.info("PositionSizer | daily loss budget exhausted")
            return 0
        risk_cents = min(risk_cents, remaining_daily_budget)

        # Cost per contract in cents
        ask_cents = max(1, int(round(kalshi_ask * 100)))
        contracts = int(math.floor(risk_cents / ask_cents))
        contracts = max(contracts, 0)

        # Tier-based size adjustment
        tier = contract.liquidity_tier
        tier_mult = 1.0
        if tier == LiquidityTier.EMPTY:
            tier_mult = 0.50
        elif tier == LiquidityTier.THIN_MARGINAL:
            tier_mult = 0.50
        elif tier == LiquidityTier.THIN_TRADEABLE:
            tier_mult = 0.75
        if tier_mult < 1.0:
            contracts = max(1, int(contracts * tier_mult)) if contracts > 0 else 0

        logger.info(
            "PositionSizer | horizon=%s tier=%s p=%.3f ask=%.2f f*=%.4f "
            "scaled=%.4f capped=%.4f risk=%dc mult=%.2f -> %d contracts",
            horizon, tier.value, p_model, kalshi_ask, full_kelly,
            scaled, capped, risk_cents, tier_mult, contracts,
        )
        return contracts
