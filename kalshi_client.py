"""
Kalshi Execution Engine — Unified for Daily and 15-minute contracts.

Async HTTP client for the Kalshi v2 API that:
1. Authenticates via RSA-PSS per-request signing.
2. Fetches active BTC-daily (KXBTC) and BTC-15m markets.
3. Reads contract order books to find best bid/ask.
4. Places limit orders for both contract types.

All prices on Kalshi are in **cents** (1-99 for YES contracts).
"""

from __future__ import annotations

import base64
import logging
import time
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlparse

import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from pydantic import BaseModel

from models import ChosenContract, ExecutionMode, LiquidityTier, OrderAction, PendingOrder, Side, TradeInstruction
from risk import PositionSizer, RiskManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KALSHI_API_BASE_PROD = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_API_BASE_DEMO = "https://demo-api.kalshi.co/trade-api/v2"


class KalshiConfig(BaseModel):
    api_base: str = KALSHI_API_BASE_PROD
    api_key_id: str = ""
    # Path to RSA private key PEM file (generated in Kalshi account settings)
    private_key_path: str = ""
    # Series ticker for 15m BTC markets
    fifteen_min_series: str = "KXBTC15M"
    # Minimum edge threshold
    min_edge: float = 0.04


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class ExecutionGuard:
    """
    Kill switch conditions beyond basic risk management.
    Tracks execution health metrics and signals when to halt or degrade.
    """

    def __init__(self) -> None:
        self.submitted_orders: int = 0       # Total orders submitted (rolling window)
        self.filled_orders: int = 0          # Total fills (rolling window)
        self.consecutive_unfilled: int = 0   # Consecutive orders without a fill
        self.dead_empty_counts: list[float] = []  # % of DEAD/EMPTY per cycle (last 5)
        self.median_spreads: list[float] = []     # Median spread per cycle (last 3)
        self._stink_only_until: float = 0.0       # Monotonic time when stink-only expires

    def record_submission(self) -> None:
        self.submitted_orders += 1
        self.consecutive_unfilled += 1

    def record_fill(self) -> None:
        self.filled_orders += 1
        self.consecutive_unfilled = 0

    def record_cycle_liquidity(self, dead_empty_pct: float, median_spread: float) -> None:
        self.dead_empty_counts.append(dead_empty_pct)
        if len(self.dead_empty_counts) > 5:
            self.dead_empty_counts.pop(0)
        self.median_spreads.append(median_spread)
        if len(self.median_spreads) > 3:
            self.median_spreads.pop(0)

    def should_halt(self) -> tuple[bool, str]:
        """Check kill switch conditions. Returns (halt, reason)."""
        # Fill rate collapse: 10 consecutive unfilled submissions
        if self.consecutive_unfilled >= 10:
            import time
            self._stink_only_until = time.monotonic() + 100  # ~10 cycles
            logger.warning(
                "KILL SWITCH | fill rate collapse: %d consecutive unfilled orders → stink-only",
                self.consecutive_unfilled,
            )
            self.consecutive_unfilled = 0  # Reset after engaging
            return False, "fill_rate_collapse_stink_only"

        # Market offline: >90% DEAD/EMPTY for 5 consecutive cycles
        if len(self.dead_empty_counts) >= 5 and all(
            pct > 0.90 for pct in self.dead_empty_counts
        ):
            return False, "market_offline_stink_only"

        # Spread blowout: median spread > 20c for 3 consecutive cycles
        if len(self.median_spreads) >= 3 and all(
            s > 20 for s in self.median_spreads
        ):
            return False, "spread_blowout_stink_only"

        return False, ""

    @property
    def is_stink_only(self) -> bool:
        """True if currently in degraded stink-only mode."""
        import time
        _, reason = self.should_halt()
        if reason.endswith("_stink_only"):
            return True
        return time.monotonic() < self._stink_only_until


class KalshiClient:
    """
    Async Kalshi API client with unified support for daily and 15m contracts.
    """

    def __init__(
        self,
        config: KalshiConfig,
        risk_manager: RiskManager,
        position_sizer: PositionSizer | None = None,
        ws=None,
    ) -> None:
        self.cfg = config
        self.risk = risk_manager
        self.sizer = position_sizer
        self._ws = ws  # KalshiWebSocket instance (optional)
        self._session: aiohttp.ClientSession | None = None
        self._private_key = None
        self._pending_orders: dict[str, PendingOrder] = {}  # order_id -> PendingOrder
        self.guard = ExecutionGuard()

        # Resting order configuration
        self.MIN_STINK_EDGE_CENTS = 8
        self.MAX_RESTING_ORDERS = 7
        self.MAX_STINK_CYCLES = 15       # ~2.5 minutes at 10s cycles
        self.PASSIVE_MAX_CYCLES = 15     # ~2.5 minutes for passive orders
        self.MODERATE_MAX_CYCLES = 6     # ~1 minute for moderate orders
        self.EXPIRY_CANCEL_SECONDS = 120  # cancel 2 min before close

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._load_private_key()
        self._session = aiohttp.ClientSession(
            headers={"Content-Type": "application/json"},
        )
        logger.info("KalshiClient started | demo=%s key_id=%s",
                    self.cfg.api_base == KALSHI_API_BASE_DEMO, self.cfg.api_key_id)

    async def stop(self) -> None:
        if self._session:
            await self._session.close()

    # ------------------------------------------------------------------
    # API Key Authentication (RSA-PSS per request)
    # ------------------------------------------------------------------

    def _load_private_key(self) -> None:
        """Load RSA private key from PEM file."""
        if not self.cfg.private_key_path:
            raise RuntimeError("private_key_path is not set in KalshiConfig")
        with open(self.cfg.private_key_path, "rb") as f:
            self._private_key = serialization.load_pem_private_key(f.read(), password=None)
        logger.info("Loaded RSA private key from %s", self.cfg.private_key_path)

    def _sign_request(self, method: str, path: str) -> dict[str, str]:
        """
        Build Kalshi API key auth headers for a single request.

        Kalshi signing spec:
            message = timestamp_ms + METHOD + /trade-api/v2<path_without_base>
            signature = base64(RSA-PSS-SHA256(message))
        """
        timestamp_ms = str(int(time.time() * 1000))
        # Extract the path portion from the full URL reliably
        # Kalshi signing requires: timestamp_ms + METHOD + /trade-api/v2/...
        api_path = urlparse(path).path  # e.g. "/trade-api/v2/portfolio/balance"
        msg = (timestamp_ms + method.upper() + api_path).encode()
        sig_bytes = self._private_key.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.cfg.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig_bytes).decode(),
        }

    def _ensure_auth(self) -> None:
        if self._private_key is None:
            raise RuntimeError("KalshiClient private key not loaded — call start() first")

    # ------------------------------------------------------------------
    # Portfolio Balance
    # ------------------------------------------------------------------

    async def fetch_balance(self) -> int:
        """
        Fetch the current portfolio balance from Kalshi in cents.
        Returns balance in cents, or -1 on failure.
        """
        self._ensure_auth()
        url = f"{self.cfg.api_base}/portfolio/balance"
        assert self._session is not None
        try:
            auth_headers = self._sign_request("GET", url)
            async with self._session.get(url, headers=auth_headers,
                                         timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                data = await resp.json()
                balance = data.get("balance", 0)
                logger.info("Kalshi balance: %d cents ($%.2f)", balance, balance / 100)
                return balance
        except Exception:
            logger.exception("Failed to fetch Kalshi balance")
            return -1

    # ------------------------------------------------------------------
    # Position Reconciliation
    # ------------------------------------------------------------------

    async def fetch_open_position_count(self) -> int:
        """
        Fetch the actual number of open positions from Kalshi.
        Returns the count, or -1 on failure.

        Uses GET /portfolio/positions?settlement_status=unsettled to count
        only positions that haven't settled yet.
        """
        self._ensure_auth()
        url = f"{self.cfg.api_base}/portfolio/positions"
        params = {"settlement_status": "unsettled", "limit": 100}
        assert self._session is not None
        try:
            auth_headers = self._sign_request("GET", url)
            async with self._session.get(url, params=params, headers=auth_headers,
                                         timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                data = await resp.json()
                positions = data.get("market_positions", data.get("positions", []))
                if positions:
                    logger.debug("Position count raw sample | %s", positions[0])
                # Count positions with non-zero holding
                # Kalshi v2 uses string fields: position_fp, market_exposure_dollars
                open_count = 0
                for p in positions:
                    position_fp = abs(float(p.get("position_fp", "0") or "0"))
                    market_exposure = abs(float(p.get("market_exposure_dollars", "0") or "0"))
                    if position_fp > 0 or market_exposure > 0:
                        open_count += 1
                logger.info(
                    "Kalshi position reconciliation: %d open positions (from %d raw)",
                    open_count, len(positions),
                )
                return open_count
        except Exception:
            logger.exception("Failed to fetch open positions from Kalshi")
            return -1

    # ------------------------------------------------------------------
    # Startup Reconciliation
    # ------------------------------------------------------------------

    async def fetch_open_positions(self) -> list[dict[str, Any]]:
        """
        Fetch full details of all open (unsettled) positions from Kalshi.
        Returns a list of position dicts, or empty list on failure.
        """
        self._ensure_auth()
        url = f"{self.cfg.api_base}/portfolio/positions"
        params = {"settlement_status": "unsettled", "limit": 100}
        assert self._session is not None
        try:
            auth_headers = self._sign_request("GET", url)
            async with self._session.get(url, params=params, headers=auth_headers,
                                         timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                data = await resp.json()
                # Log raw response keys and first position for field discovery
                logger.info(
                    "Positions API raw | keys=%s count=%d",
                    list(data.keys()),
                    len(data.get("market_positions", data.get("positions", []))),
                )
                positions = data.get("market_positions", data.get("positions", []))
                if positions:
                    logger.info("Positions API sample | %s", positions[0])
                # Accept any position with non-zero quantity on either side
                # Kalshi v2 uses string fields: position_fp, total_traded_dollars,
                # market_exposure_dollars, resting_orders_count
                result = []
                for p in positions:
                    position_fp = abs(float(p.get("position_fp", "0") or "0"))
                    market_exposure = abs(float(p.get("market_exposure_dollars", "0") or "0"))
                    total_traded = abs(float(p.get("total_traded_dollars", "0") or "0"))
                    resting = p.get("resting_orders_count", 0) or 0
                    # Also check legacy integer fields just in case
                    yes_qty = abs(p.get("yes_sub_total", 0) or 0)
                    no_qty = abs(p.get("no_sub_total", 0) or 0)
                    if position_fp > 0 or market_exposure > 0 or total_traded > 0 or resting > 0 or yes_qty > 0 or no_qty > 0:
                        result.append(p)
                logger.info("Positions API | %d raw -> %d with nonzero qty", len(positions), len(result))
                return result
        except Exception:
            logger.exception("Failed to fetch open positions from Kalshi")
            return []

    async def fetch_resting_orders(self) -> list[dict[str, Any]]:
        """
        Fetch all resting (open) orders from Kalshi.
        Returns a list of order dicts, or empty list on failure.
        """
        self._ensure_auth()
        url = f"{self.cfg.api_base}/portfolio/orders"
        params = {"status": "resting", "limit": 100}
        assert self._session is not None
        try:
            auth_headers = self._sign_request("GET", url)
            async with self._session.get(url, params=params, headers=auth_headers,
                                         timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                data = await resp.json()
                # Log raw response for field discovery
                logger.info(
                    "Orders API raw | status=%d keys=%s count=%d",
                    resp.status, list(data.keys()),
                    len(data.get("orders", [])),
                )
                orders = data.get("orders", [])
                if orders:
                    logger.info("Orders API sample | %s", orders[0])
                else:
                    # Try without status filter to discover what Kalshi returns
                    logger.info("No orders with status=resting — trying unfiltered fetch for diagnostics")
                    auth_headers2 = self._sign_request("GET", url)
                    async with self._session.get(url, params={"limit": 20}, headers=auth_headers2,
                                                 timeout=aiohttp.ClientTimeout(total=10)) as resp2:
                        if resp2.status == 200:
                            data2 = await resp2.json()
                            all_orders = data2.get("orders", [])
                            if all_orders:
                                statuses = [o.get("status", "?") for o in all_orders[:10]]
                                logger.info(
                                    "Unfiltered orders: %d total, statuses=%s, sample=%s",
                                    len(all_orders), statuses, all_orders[0],
                                )
                                # Return orders that look like they're still open
                                orders = [
                                    o for o in all_orders
                                    if o.get("status") in ("resting", "open", "pending", "active")
                                    or float(o.get("remaining_count_fp", "0") or "0") > 0
                                ]
                            else:
                                logger.info("Unfiltered orders also empty — keys=%s", list(data2.keys()))
                return orders
        except Exception:
            logger.exception("Failed to fetch resting orders from Kalshi")
            return []

    async def reconcile_on_startup(self, router) -> None:
        """
        Query Kalshi for the true live account state and rebuild all
        in-memory tracking before the first trading cycle.

        Steps:
        1. Fetch balance, open positions, and resting orders concurrently.
        2. Rebuild router active positions from live positions.
        3. Rebuild _pending_orders from live resting orders.
        4. Sync risk manager counters (equity, position count, resting risk).
        5. Log a clear reconciliation summary.
        """
        import asyncio

        logger.info("=" * 60)
        logger.info("STARTUP RECONCILIATION — querying Kalshi for live state")
        logger.info("=" * 60)

        # 1. Fetch all state concurrently
        balance_result, positions_result, orders_result = await asyncio.gather(
            self.fetch_balance(),
            self.fetch_open_positions(),
            self.fetch_resting_orders(),
            return_exceptions=True,
        )

        balance = balance_result if isinstance(balance_result, int) else -1
        positions = positions_result if isinstance(positions_result, list) else []
        resting_orders = orders_result if isinstance(orders_result, list) else []

        if isinstance(balance_result, Exception):
            logger.error("Reconciliation: balance fetch failed: %s", balance_result)
        if isinstance(positions_result, Exception):
            logger.error("Reconciliation: positions fetch failed: %s", positions_result)
        if isinstance(orders_result, Exception):
            logger.error("Reconciliation: orders fetch failed: %s", orders_result)

        # 2. Sync equity
        if balance > 0:
            self.risk.sync_equity(balance)
        else:
            logger.warning(
                "Reconciliation: could not fetch balance — using initial equity %d cents",
                self.risk.cfg.initial_equity_cents,
            )

        # 3. Rebuild router active positions from live Kalshi positions
        #    Kalshi v2 fields: position_fp (string, positive=YES, negative=NO),
        #    market_exposure_dollars (string), resting_orders_count (int)
        recovered_positions = 0
        for pos in positions:
            ticker = pos.get("ticker", "")
            if not ticker:
                continue
            # position_fp: positive = long YES, negative = long NO
            position_fp = float(pos.get("position_fp", "0") or "0")
            if position_fp > 0:
                side = Side.YES
            elif position_fp < 0:
                side = Side.NO
            else:
                # Zero position but nonzero exposure/resting — register from resting orders later
                continue
            router.register_position(ticker, side)
            recovered_positions += 1
            logger.info(
                "Reconciliation: recovered position | %s %s qty=%.0f exposure=$%s",
                side.value.upper(), ticker, abs(position_fp),
                pos.get("market_exposure_dollars", "0"),
            )

        # 4. Sync position count (only positions with nonzero holdings)
        self.risk.sync_position_count(recovered_positions)

        # 5. Rebuild _pending_orders from live resting orders
        self._pending_orders.clear()
        self.risk.resting_order_count = 0
        self.risk.resting_risk_cents = 0
        orphaned_orders = 0

        for order in resting_orders:
            order_id = order.get("order_id", "")
            if not order_id:
                continue
            ticker = order.get("ticker", "")
            side_str = order.get("side", "yes")
            side = Side.YES if side_str == "yes" else Side.NO
            action_str = order.get("action", "buy")

            # Extract price (Kalshi v2: yes_price_dollars / no_price_dollars as strings)
            yes_price_str = order.get("yes_price_dollars") or order.get("yes_price") or "0"
            no_price_str = order.get("no_price_dollars") or order.get("no_price") or "0"
            yes_price_cents = int(round(float(yes_price_str) * 100)) if float(yes_price_str) < 1.01 else int(yes_price_str)
            no_price_cents = int(round(float(no_price_str) * 100)) if float(no_price_str) < 1.01 else int(no_price_str)
            limit_cents = yes_price_cents if side == Side.YES else no_price_cents
            if limit_cents <= 0:
                limit_cents = 1

            # Extract quantity (Kalshi v2: remaining_count_fp / initial_count_fp as strings)
            remaining_str = order.get("remaining_count_fp") or order.get("remaining_count") or "0"
            initial_str = order.get("initial_count_fp") or order.get("count") or "0"
            remaining = int(float(remaining_str))
            if remaining <= 0:
                remaining = int(float(initial_str))

            if remaining <= 0:
                continue

            # Parse expiry from order if available
            expiry_utc = None
            expiry_str = order.get("expiration_time") or order.get("close_time")
            if expiry_str:
                try:
                    expiry_utc = datetime.fromisoformat(
                        expiry_str.replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass

            # Determine horizon from ticker
            horizon: Literal["daily", "15m"] = "daily"
            if self.cfg.fifteen_min_series in ticker:
                horizon = "15m"

            created_str = order.get("created_time")
            placed_at = datetime.now(timezone.utc)
            if created_str:
                try:
                    placed_at = datetime.fromisoformat(
                        created_str.replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass

            pending = PendingOrder(
                order_id=order_id,
                ticker=ticker,
                side=side,
                action=OrderAction.BUY if action_str == "buy" else OrderAction.SELL,
                limit_price_cents=max(1, min(99, limit_cents)),
                contracts=remaining,
                horizon=horizon,
                liquidity_tier=LiquidityTier.EMPTY,
                model_p_at_placement=0.0,  # unknown from prior session
                edge_at_placement=0.0,     # unknown from prior session
                placed_at=placed_at,
                expiry_utc=expiry_utc,
                cycles_alive=0,
                status="resting",
            )
            self._pending_orders[order_id] = pending

            # Track resting risk
            cost = pending.limit_price_cents * pending.contracts
            self.risk.notify_resting_order_placed(cost)

            # Register in router so we don't trade opposite side
            router.register_position(ticker, side)
            orphaned_orders += 1

            logger.info(
                "Reconciliation: recovered resting order | id=%s %s %s %s x%d @%dc",
                order_id, side.value.upper(), ticker, horizon,
                remaining, limit_cents,
            )

        # 6. Cancel excess resting orders down to MAX_RESTING_ORDERS cap
        excess = self.resting_count - self.MAX_RESTING_ORDERS
        if excess > 0:
            logger.info(
                "Reconciliation: %d resting orders exceed cap of %d — cancelling %d oldest",
                self.resting_count, self.MAX_RESTING_ORDERS, excess,
            )
            # Sort resting orders: prioritise cancelling oldest (lowest edge, oldest placed_at)
            resting_list = [
                (oid, p) for oid, p in self._pending_orders.items()
                if p.status == "resting"
            ]
            resting_list.sort(key=lambda x: (x[1].edge_at_placement, x[1].placed_at))

            cancelled_excess = 0
            for oid, p in resting_list:
                if cancelled_excess >= excess:
                    break
                success = await self.cancel_order(oid)
                if success:
                    p.status = "cancelled"
                    cost = p.limit_price_cents * p.contracts
                    self.risk.notify_resting_order_resolved(cost, was_filled=False)
                    # Remove lockout for this ticker if no other resting order on it
                    has_other = any(
                        o.ticker == p.ticker and o.status == "resting" and o.order_id != oid
                        for o in self._pending_orders.values()
                    )
                    if not has_other:
                        router.clear_position(p.ticker)
                    cancelled_excess += 1
                    logger.info(
                        "Reconciliation: cancelled excess order | id=%s %s %s x%d @%dc",
                        oid, p.side.value.upper(), p.ticker,
                        p.contracts, p.limit_price_cents,
                    )
            logger.info(
                "Reconciliation: cancelled %d excess orders — %d resting remain",
                cancelled_excess, self.resting_count,
            )

        # 7. Log summary
        logger.info("=" * 60)
        logger.info("RECONCILIATION COMPLETE")
        logger.info(
            "  Balance:          %d cents ($%.2f)",
            balance if balance > 0 else self.risk.equity_cents,
            (balance if balance > 0 else self.risk.equity_cents) / 100,
        )
        logger.info("  Open positions:   %d", recovered_positions)
        logger.info("  Resting orders:   %d recovered, %d kept (cap=%d)",
                     orphaned_orders, self.resting_count, self.MAX_RESTING_ORDERS)
        logger.info("  Position lockouts: %d", len(router._active_positions))
        logger.info(
            "  Resting risk:     %d cents ($%.2f)",
            self.risk.resting_risk_cents, self.risk.resting_risk_cents / 100,
        )
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Market Discovery — Unified
    # ------------------------------------------------------------------

    async def _fetch_markets_by_series(self, series_ticker: str) -> list[dict[str, Any]]:
        """Fetch active markets for a given series ticker."""
        self._ensure_auth()
        url = f"{self.cfg.api_base}/markets"
        params = {
            "series_ticker": series_ticker,
            "status": "open",
            "limit": 50,
        }
        assert self._session is not None
        try:
            auth_headers = self._sign_request("GET", url)
            async with self._session.get(url, params=params, headers=auth_headers,
                                         timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                data = await resp.json()
                markets = data.get("markets", [])
                logger.info("Fetched %d markets for series %s", len(markets), series_ticker)
                return markets
        except Exception:
            logger.exception("Failed to fetch markets for series %s", series_ticker)
            return []

    async def get_active_15m_markets(self) -> list[dict[str, Any]]:
        """
        Fetch 15m markets only.
        If a WebSocket is attached, enriches listing prices with live data
        and updates subscriptions.
        """
        markets = await self._fetch_markets_by_series(self.cfg.fifteen_min_series)

        # Enrich with live WS data and manage subscriptions
        if self._ws is not None:
            tickers = [m.get("ticker", "") for m in markets if m.get("ticker")]
            await self._ws.update_subscriptions(tickers)
            markets = self._enrich_markets_with_ws(markets)

        return markets

    def _enrich_markets_with_ws(self, markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Overwrite listing prices with live WebSocket bid/ask where available.
        Also sets a '_has_ws_depth' flag for the router to check.
        """
        if self._ws is None:
            return markets
        for market in markets:
            ticker = market.get("ticker", "")
            if not ticker:
                continue
            prices = self._ws.get_best_prices(ticker)
            if prices is not None and prices["has_depth"]:
                # Overwrite stale listing prices with live data
                if prices["yes_ask_cents"] is not None:
                    market["yes_ask_dollars"] = f"{prices['yes_ask_cents'] / 100:.2f}"
                if prices["yes_bid_cents"] is not None:
                    market["yes_bid_dollars"] = f"{prices['yes_bid_cents'] / 100:.2f}"
                market["_has_ws_depth"] = True
                market["_ws_liquidity_tier"] = prices.get("liquidity_tier", "thin")
                market["_ws_total_depth"] = prices.get("total_depth", 0)
            else:
                market["_has_ws_depth"] = False
                market["_ws_liquidity_tier"] = "empty"
                market["_ws_total_depth"] = 0
        return markets

    async def get_market_orderbook(self, ticker: str) -> dict[str, Any]:
        """
        Get the order book for a single contract ticker.
        Prefers WebSocket local book if available, falls back to REST.
        """
        # Try WebSocket local book first (instant, no network)
        if self._ws is not None:
            ws_book = self._ws.get_orderbook_rest_format(ticker)
            if ws_book is not None:
                logger.debug("Using WS local book for %s", ticker)
                return ws_book

        # Fall back to REST
        self._ensure_auth()
        url = f"{self.cfg.api_base}/markets/{ticker}/orderbook"
        assert self._session is not None
        try:
            auth_headers = self._sign_request("GET", url)
            async with self._session.get(url, headers=auth_headers,
                                         timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                data = await resp.json()
                orderbook = data.get("orderbook", {})
                if not orderbook:
                    logger.debug(
                        "Orderbook API returned empty for %s — raw keys: %s",
                        ticker, list(data.keys()),
                    )
                return orderbook
        except Exception:
            logger.exception("Failed to fetch orderbook for %s", ticker)
            return {}

    # ------------------------------------------------------------------
    # Best-Ask Extraction
    # ------------------------------------------------------------------

    @staticmethod
    def best_yes_ask(orderbook: dict[str, Any]) -> int | None:
        """Return the lowest YES ask price (cents) from the Kalshi orderbook."""
        yes_levels = orderbook.get("yes", [])
        if not yes_levels:
            logger.debug("Orderbook has no YES levels. Keys: %s", list(orderbook.keys()))
            return None
        # Log first level to verify field names on first encounter
        logger.debug("Orderbook YES level sample: %s", yes_levels[0])
        prices = [lvl["price"] for lvl in yes_levels if lvl.get("delta", 0) > 0]
        return min(prices) if prices else None

    @staticmethod
    def best_no_ask(orderbook: dict[str, Any]) -> int | None:
        """Return the lowest NO ask price (cents) from the Kalshi orderbook."""
        no_levels = orderbook.get("no", [])
        if not no_levels:
            logger.debug("Orderbook has no NO levels. Keys: %s", list(orderbook.keys()))
            return None
        logger.debug("Orderbook NO level sample: %s", no_levels[0])
        prices = [lvl["price"] for lvl in no_levels if lvl.get("delta", 0) > 0]
        return min(prices) if prices else None

    # ------------------------------------------------------------------
    # Edge Calculation
    # ------------------------------------------------------------------

    def calculate_edge(self, model_prob: float, ask_cents: int, side: Side = Side.YES) -> float:
        """
        Edge = model probability - implied market probability.
        For YES contracts: implied = ask / 100.
        """
        if side == Side.YES:
            implied = ask_cents / 100.0
        else:
            implied = (100 - ask_cents) / 100.0
        return model_prob - implied

    # ------------------------------------------------------------------
    # Order Lifecycle Management
    # ------------------------------------------------------------------

    async def check_order_status(self, order_id: str) -> dict[str, Any] | None:
        """GET /portfolio/orders/{order_id} — check if a resting order has filled."""
        self._ensure_auth()
        url = f"{self.cfg.api_base}/portfolio/orders/{order_id}"
        assert self._session is not None
        try:
            auth_headers = self._sign_request("GET", url)
            async with self._session.get(url, headers=auth_headers,
                                         timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("order", data)
                else:
                    body = await resp.text()
                    logger.warning("Order status check failed [%d]: %s", resp.status, body)
                    return None
        except Exception:
            logger.exception("Failed to check order status for %s", order_id)
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """DELETE /portfolio/orders/{order_id} — cancel a resting order."""
        self._ensure_auth()
        url = f"{self.cfg.api_base}/portfolio/orders/{order_id}"
        assert self._session is not None
        try:
            auth_headers = self._sign_request("DELETE", url)
            async with self._session.delete(url, headers=auth_headers,
                                            timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status in (200, 204):
                    logger.info("Order cancelled | id=%s", order_id)
                    return True
                elif resp.status == 404:
                    # Already filled or cancelled
                    logger.info("Order cancel 404 (already resolved) | id=%s", order_id)
                    return True
                else:
                    body = await resp.text()
                    logger.warning("Order cancel failed [%d]: %s", resp.status, body)
                    return False
        except Exception:
            logger.exception("Failed to cancel order %s", order_id)
            return False

    async def cancel_all_pending(self, reason: str = "") -> int:
        """Cancel all resting orders. Returns count of successfully cancelled."""
        cancelled = 0
        for order_id, pending in list(self._pending_orders.items()):
            if pending.status == "resting":
                success = await self.cancel_order(order_id)
                if success:
                    pending.status = "cancelled"
                    cost = pending.limit_price_cents * pending.contracts
                    self.risk.notify_resting_order_resolved(cost, was_filled=False)
                    cancelled += 1
        if cancelled:
            logger.info("Cancelled %d pending orders | reason=%s", cancelled, reason)
        return cancelled

    async def manage_pending_orders(self) -> tuple[int, int, int]:
        """
        Lifecycle management for all pending orders. Called each cycle.
        Returns (filled_count, cancelled_count, still_resting_count).
        """
        now = datetime.now(timezone.utc)
        filled = 0
        cancelled = 0
        still_resting = 0

        for order_id, pending in list(self._pending_orders.items()):
            if pending.status != "resting":
                continue

            pending.cycles_alive += 1

            # 1. Check if filled (earlier for passive/moderate, 3 cycles for stink)
            min_check_cycles = 1 if pending.execution_mode in ("passive", "moderate") else 3
            if pending.cycles_alive >= min_check_cycles:
                order_data = await self.check_order_status(order_id)
                if order_data is not None:
                    remaining = order_data.get("remaining_count", -1)
                    status = order_data.get("status", "")
                    if remaining == 0 or status in ("executed", "filled"):
                        pending.status = "filled"
                        cost = pending.limit_price_cents * pending.contracts
                        self.risk.notify_resting_order_resolved(cost, was_filled=True)
                        self.guard.record_fill()
                        filled += 1
                        logger.info(
                            "FILLED | %s %s %s x%d @%dc | rested %d cycles",
                            pending.side.value, pending.ticker,
                            pending.horizon, pending.contracts,
                            pending.limit_price_cents, pending.cycles_alive,
                        )
                        continue
                    elif status in ("canceled", "cancelled"):
                        pending.status = "cancelled"
                        cost = pending.limit_price_cents * pending.contracts
                        self.risk.notify_resting_order_resolved(cost, was_filled=False)
                        cancelled += 1
                        continue

            # 2. Cancel if approaching expiry (shorter buffer for 15m markets)
            cancel_secs = 30 if pending.horizon == "15m" else self.EXPIRY_CANCEL_SECONDS
            if pending.expiry_utc and (pending.expiry_utc - now).total_seconds() < cancel_secs:
                await self.cancel_order(order_id)
                pending.status = "cancelled"
                cost = pending.limit_price_cents * pending.contracts
                self.risk.notify_resting_order_resolved(cost, was_filled=False)
                cancelled += 1
                logger.info("CANCEL (near expiry) | %s | %ds to close",
                            pending.ticker,
                            int((pending.expiry_utc - now).total_seconds()))
                continue

            # 2b. Bid pegging: reprice if best bid improved (THIN_TRADEABLE only)
            repeg_price = await self._should_repeg(pending)
            if repeg_price is not None:
                result = await self._improve_resting_order(order_id, pending, repeg_price)
                if result:
                    still_resting += 1
                else:
                    cancelled += 1
                continue

            # 3. Cancel if max cycles exceeded (mode + tier dependent patience)
            if pending.execution_mode == "moderate":
                # THIN_TRADEABLE midpoint sniper: shorter patience (4 cycles)
                if pending.liquidity_tier == LiquidityTier.THIN_TRADEABLE:
                    max_cycles = 4
                else:
                    max_cycles = self.MODERATE_MAX_CYCLES
            elif pending.execution_mode == "passive":
                # THIN_MARGINAL conservative: shorter patience (6 cycles)
                if pending.liquidity_tier == LiquidityTier.THIN_MARGINAL:
                    max_cycles = 6
                else:
                    max_cycles = self.PASSIVE_MAX_CYCLES
            else:
                max_cycles = self.MAX_STINK_CYCLES

            if pending.cycles_alive > max_cycles:
                await self.cancel_order(order_id)
                pending.status = "cancelled"
                cost = pending.limit_price_cents * pending.contracts
                self.risk.notify_resting_order_resolved(cost, was_filled=False)
                cancelled += 1
                logger.info("CANCEL (patience expired) | %s | alive %d/%d cycles (mode=%s)",
                            pending.ticker, pending.cycles_alive, max_cycles,
                            pending.execution_mode)
                continue

            still_resting += 1

        # Prune resolved orders (keep last 50 for audit)
        resolved = [oid for oid, p in self._pending_orders.items()
                     if p.status in ("filled", "cancelled", "expired")]
        if len(resolved) > 50:
            for oid in sorted(resolved, key=lambda o: self._pending_orders[o].placed_at)[:len(resolved) - 50]:
                del self._pending_orders[oid]

        return filled, cancelled, still_resting

    @property
    def resting_count(self) -> int:
        """Number of currently resting orders."""
        return sum(1 for p in self._pending_orders.values() if p.status == "resting")

    # ------------------------------------------------------------------
    # Passive Limit Execution (DEEP/THIN tier — resting-order-first)
    # ------------------------------------------------------------------

    def _compute_passive_price(
        self,
        contract: ChosenContract,
        orderbook: dict[str, Any],
        moderate: bool = False,
    ) -> int | None:
        """
        Compute limit price for a passive order that posts inside the spread.

        PASSIVE: bid+1 (narrow spread) or midpoint (wide spread) — maximize edge.
        MODERATE: mid+1 (narrow) or ask-1 (wide) — balance edge vs fill speed.

        Returns price in cents, or None if no valid passive price exists.
        """
        if contract.side == Side.YES:
            best_ask = self.best_yes_ask(orderbook)
            best_bid = self.best_yes_bid(orderbook)
        else:
            best_ask = self.best_no_ask(orderbook)
            yes_ask = self.best_yes_ask(orderbook)
            best_bid = (100 - yes_ask) if yes_ask is not None else None

        if best_ask is None:
            return None

        if best_bid is not None and best_ask > best_bid:
            spread_cents = best_ask - best_bid
            if moderate:
                if spread_cents <= 3:
                    price = best_ask                    # tiny spread, just take
                elif spread_cents <= 8:
                    price = (best_ask + best_bid + 1) // 2  # just above mid
                else:
                    price = best_ask - 1                # 1 tick below ask
            else:
                if spread_cents <= 5:
                    price = best_bid + 1                # 1 tick above bid
                else:
                    price = (best_ask + best_bid) // 2  # midpoint
        else:
            price = best_ask - 1 if moderate else max(1, best_ask - 2)

        # Edge floor: preserve minimum edge
        model_p_cents = int(contract.p_model * 100)
        min_edge_cents = max(1, int(self.cfg.min_edge * 100))
        max_price = model_p_cents - min_edge_cents
        price = min(price, max_price)

        price = max(1, min(99, price))

        edge = contract.p_model - (price / 100.0)
        if edge < self.cfg.min_edge:
            logger.debug(
                "Passive price %dc yields insufficient edge %.4f for %s",
                price, edge, contract.ticker,
            )
            return None

        return price

    async def execute_passive_limit(
        self,
        contract: ChosenContract,
        num_contracts: int,
        execution_mode: ExecutionMode,
    ) -> PendingOrder | None:
        """
        Post a limit order inside the spread as a resting passive order.
        Tracked as PendingOrder — fill is detected in manage_pending_orders.
        """
        if num_contracts <= 0:
            return None

        # Duplicate check: skip if already resting on this ticker
        for p in self._pending_orders.values():
            if p.ticker == contract.ticker and p.status == "resting":
                logger.debug("Passive skip | already resting on %s", contract.ticker)
                return None

        # Resting order cap (shared with stink bids) — try rotation
        if self.resting_count >= self.MAX_RESTING_ORDERS:
            worst_oid, worst_edge = None, float("inf")
            for oid, p in self._pending_orders.items():
                if p.status == "resting" and p.edge_at_placement < worst_edge:
                    worst_oid, worst_edge = oid, p.edge_at_placement
            if worst_oid and contract.edge > worst_edge + 0.005:
                worst = self._pending_orders[worst_oid]
                logger.info(
                    "ROTATE | cancel %s (edge=%.4f) for passive %s (edge=%.4f)",
                    worst.ticker, worst_edge, contract.ticker, contract.edge,
                )
                await self.cancel_order(worst_oid)
                worst.status = "cancelled"
                cost = worst.limit_price_cents * worst.contracts
                self.risk.notify_resting_order_resolved(cost, was_filled=False)
            else:
                logger.info(
                    "Passive skip | %d resting, no worse-edge to rotate",
                    self.resting_count,
                )
                return None

        # Fetch orderbook for pricing
        orderbook = await self.get_market_orderbook(contract.ticker)
        if not orderbook:
            logger.warning("No orderbook for passive limit on %s", contract.ticker)
            return None

        moderate = (execution_mode == ExecutionMode.MODERATE)
        limit_cents = self._compute_passive_price(contract, orderbook, moderate=moderate)

        if limit_cents is None:
            logger.info(
                "No valid passive price for %s (mode=%s)",
                contract.ticker, execution_mode.value,
            )
            return None

        # Risk gate (resting orders use separate gate — don't block filled positions)
        total_cost = limit_cents * num_contracts
        allowed, reason = self.risk.passes_risk_checks_for_stink_bid(
            contract.edge, total_cost,
        )
        if not allowed:
            logger.info("Passive limit blocked by risk: %s", reason)
            return None

        instruction = TradeInstruction(
            ticker=contract.ticker,
            action=OrderAction.BUY,
            side=contract.side,
            contracts=num_contracts,
            limit_price_cents=limit_cents,
            model_probability=contract.p_model,
            edge=contract.p_model - (limit_cents / 100.0),
            horizon=contract.horizon,
            liquidity_tier=contract.liquidity_tier,
            metadata={
                "strategy": f"passive_{execution_mode.value}",
                "original_ask": (
                    self.best_yes_ask(orderbook)
                    if contract.side == Side.YES
                    else self.best_no_ask(orderbook)
                ),
            },
        )

        logger.info(
            "PASSIVE LIMIT | %s %s %s x%d @%dc (mode=%s) | edge=%.4f model_p=%.4f",
            instruction.side.value, contract.ticker, contract.horizon,
            num_contracts, limit_cents, execution_mode.value,
            instruction.edge, contract.p_model,
        )

        result = await self._submit_order(instruction)
        if result:
            order_id = result.get("order", {}).get("order_id", "unknown")
            strike = contract.metadata.get("strike") if contract.metadata else None

            # Store reference bid for bid pegging (THIN_TRADEABLE)
            if contract.side == Side.YES:
                ref_bid = self.best_yes_bid(orderbook)
            else:
                ya = self.best_yes_ask(orderbook)
                ref_bid = (100 - ya) if ya is not None else None

            pending = PendingOrder(
                order_id=order_id,
                ticker=contract.ticker,
                side=contract.side,
                limit_price_cents=limit_cents,
                contracts=num_contracts,
                horizon=contract.horizon,
                liquidity_tier=contract.liquidity_tier,
                model_p_at_placement=contract.p_model,
                edge_at_placement=contract.p_model - (limit_cents / 100.0),
                expiry_utc=contract.close_time,
                execution_mode=execution_mode.value,
                strike=strike,
                metadata={
                    "reference_bid": ref_bid,
                    "improvements": 0,
                },
            )
            self._pending_orders[order_id] = pending
            self.risk.notify_resting_order_placed(total_cost)
            return pending
        return None

    # ------------------------------------------------------------------
    # Stink Bid Execution (EMPTY tier)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stink_price(model_p_cents: int, min_edge_cents: int, level: int = 1) -> int:
        """
        Compute stink bid price with adaptive edge buffering.

        Level 1 (aggressive): model_p - 2.0 * min_edge (higher fill chance)
        Level 2 (deep value): model_p - 2.5 * min_edge (better edge if filled)
        """
        buffer_mult = 2.0 if level == 1 else 2.5
        price = model_p_cents - int(min_edge_cents * buffer_mult)
        return max(1, min(99, price))

    async def execute_stink_bid(
        self,
        contract: ChosenContract,
        num_contracts: int,
    ) -> PendingOrder | None:
        """
        Place 1-2 limit orders on an empty book at model-derived prices.
        Level 1: aggressive stink (2x min_edge buffer)
        Level 2: deep value stink (2.5x min_edge buffer) — only if room for 2 orders
        """
        if num_contracts <= 0:
            return None

        # Don't place if too many pending orders already — try rotation
        if self.resting_count >= self.MAX_RESTING_ORDERS:
            worst_oid, worst_edge = None, float("inf")
            for oid, p in self._pending_orders.items():
                if p.status == "resting" and p.edge_at_placement < worst_edge:
                    worst_oid, worst_edge = oid, p.edge_at_placement
            if worst_oid and contract.edge > worst_edge + 0.005:
                worst = self._pending_orders[worst_oid]
                logger.info(
                    "ROTATE | cancel %s (edge=%.4f) for %s (edge=%.4f)",
                    worst.ticker, worst_edge, contract.ticker, contract.edge,
                )
                await self.cancel_order(worst_oid)
                worst.status = "cancelled"
                cost = worst.limit_price_cents * worst.contracts
                self.risk.notify_resting_order_resolved(cost, was_filled=False)
            else:
                logger.info("Stink bid skipped | %d resting, no better edge to rotate", self.resting_count)
                return None

        # Don't duplicate: skip if we already have a resting order on this ticker
        for p in self._pending_orders.values():
            if p.ticker == contract.ticker and p.status == "resting":
                logger.debug("Stink bid skipped | already resting on %s", contract.ticker)
                return None

        model_p_cents = int(contract.p_model * 100)
        min_edge_cents = max(1, int(self.cfg.min_edge * 100))

        # Level 1: aggressive stink
        limit_cents_1 = self._compute_stink_price(model_p_cents, min_edge_cents, level=1)
        edge_1 = contract.p_model - (limit_cents_1 / 100.0)
        if edge_1 < self.cfg.min_edge:
            # Fallback to original pricing
            limit_cents_1 = max(1, min(99, int(contract.kalshi_ask * 100)))
            edge_1 = contract.p_model - (limit_cents_1 / 100.0)

        # Risk gate for level 1
        total_cost_1 = limit_cents_1 * num_contracts
        allowed, reason = self.risk.passes_risk_checks_for_stink_bid(edge_1, total_cost_1)
        if not allowed:
            logger.info("Stink bid blocked by risk: %s", reason)
            return None

        # Determine if we can place a dual-level stink
        limit_cents_2 = self._compute_stink_price(model_p_cents, min_edge_cents, level=2)
        edge_2 = contract.p_model - (limit_cents_2 / 100.0)
        can_dual = (
            limit_cents_2 != limit_cents_1  # Different prices
            and limit_cents_2 < limit_cents_1  # Level 2 is deeper value
            and edge_2 >= self.cfg.min_edge
            and self.resting_count + 1 < self.MAX_RESTING_ORDERS  # Room for 2 orders
            and num_contracts > 1
        )

        if can_dual:
            n1 = max(1, int(num_contracts * 0.6))
            n2 = max(1, num_contracts - n1)
        else:
            n1 = num_contracts
            n2 = 0

        # Submit level 1
        instruction_1 = TradeInstruction(
            ticker=contract.ticker,
            action=OrderAction.BUY,
            side=contract.side,
            contracts=n1,
            limit_price_cents=limit_cents_1,
            model_probability=contract.p_model,
            edge=edge_1,
            horizon=contract.horizon,
            liquidity_tier=LiquidityTier.EMPTY,
            metadata={"strategy": "stink_bid_L1"},
        )

        logger.info(
            "STINK BID L1 | %s %s %s x%d @%dc | edge=%.4f model_p=%.4f%s",
            instruction_1.side.value, contract.ticker, contract.horizon,
            n1, limit_cents_1, edge_1, contract.p_model,
            f" (dual: L2 @{limit_cents_2}c x{n2})" if can_dual else "",
        )

        first_pending: PendingOrder | None = None
        result = await self._submit_order(instruction_1)
        if result:
            order_id = result.get("order", {}).get("order_id", "unknown")
            first_pending = PendingOrder(
                order_id=order_id,
                ticker=contract.ticker,
                side=contract.side,
                limit_price_cents=limit_cents_1,
                contracts=n1,
                horizon=contract.horizon,
                liquidity_tier=LiquidityTier.EMPTY,
                model_p_at_placement=contract.p_model,
                edge_at_placement=edge_1,
                expiry_utc=contract.close_time,
                metadata={"improvements": 0},
            )
            self._pending_orders[order_id] = first_pending
            self.risk.notify_resting_order_placed(limit_cents_1 * n1)

        # Submit level 2 (deep value) if dual-level
        if can_dual and n2 > 0 and first_pending is not None:
            total_cost_2 = limit_cents_2 * n2
            allowed2, reason2 = self.risk.passes_risk_checks_for_stink_bid(edge_2, total_cost_2)
            if allowed2:
                instruction_2 = TradeInstruction(
                    ticker=contract.ticker,
                    action=OrderAction.BUY,
                    side=contract.side,
                    contracts=n2,
                    limit_price_cents=limit_cents_2,
                    model_probability=contract.p_model,
                    edge=edge_2,
                    horizon=contract.horizon,
                    liquidity_tier=LiquidityTier.EMPTY,
                    metadata={"strategy": "stink_bid_L2"},
                )
                result2 = await self._submit_order(instruction_2)
                if result2:
                    oid2 = result2.get("order", {}).get("order_id", "unknown")
                    pending2 = PendingOrder(
                        order_id=oid2,
                        ticker=contract.ticker,
                        side=contract.side,
                        limit_price_cents=limit_cents_2,
                        contracts=n2,
                        horizon=contract.horizon,
                        liquidity_tier=LiquidityTier.EMPTY,
                        model_p_at_placement=contract.p_model,
                        edge_at_placement=edge_2,
                        expiry_utc=contract.close_time,
                        metadata={"improvements": 0},
                    )
                    self._pending_orders[oid2] = pending2
                    self.risk.notify_resting_order_placed(total_cost_2)
                    logger.info(
                        "STINK BID L2 | %s %s x%d @%dc | edge=%.4f",
                        contract.side.value, contract.ticker, n2, limit_cents_2, edge_2,
                    )

        return first_pending

    # ------------------------------------------------------------------
    # Thin Book Execution (THIN tier)
    # ------------------------------------------------------------------

    async def _execute_thin_book(
        self,
        contract: ChosenContract,
        num_contracts: int,
    ) -> TradeInstruction | None:
        """For THIN books: post limit at midpoint of spread."""
        orderbook = await self.get_market_orderbook(contract.ticker)
        if not orderbook:
            return None

        if contract.side == Side.YES:
            ask = self.best_yes_ask(orderbook)
            bid = self.best_yes_bid(orderbook)
        else:
            ask = self.best_no_ask(orderbook)
            bid = None

        if ask is None:
            logger.warning("No live ask for thin book %s — skipping", contract.ticker)
            return None

        # Calculate midpoint or improve by 1 tick inside the spread
        if bid is not None and ask > bid:
            mid = (ask + bid) // 2
            limit_cents = max(bid + 1, mid)
        else:
            limit_cents = ask - 1 if ask > 1 else ask

        limit_cents = max(1, min(99, limit_cents))

        # Verify edge at our limit price
        live_implied = limit_cents / 100.0
        live_edge = contract.p_model - live_implied
        if live_edge < contract.edge * 0.5:
            logger.info("Thin book edge deteriorated for %s: %.4f", contract.ticker, live_edge)
            return None

        instruction = TradeInstruction(
            ticker=contract.ticker,
            action=OrderAction.BUY,
            side=contract.side,
            contracts=num_contracts,
            limit_price_cents=limit_cents,
            model_probability=contract.p_model,
            edge=live_edge,
            horizon=contract.horizon,
            liquidity_tier=contract.liquidity_tier,
            metadata={"strategy": "post_inside", "original_ask": ask},
        )

        logger.info(
            "THIN BOOK | %s %s %s x%d @%dc (ask was %dc) | edge=%.4f",
            instruction.side.value, contract.ticker, contract.horizon,
            num_contracts, limit_cents, ask, live_edge,
        )

        result = await self._submit_order(instruction)
        if result:
            self.risk.notify_position_opened()
            return instruction
        # Submission failed — return None so caller does NOT register a phantom position
        return None

    @staticmethod
    def best_yes_bid(orderbook: dict[str, Any]) -> int | None:
        """Return the highest YES bid price (cents).

        In Kalshi binary markets, YES bid = 100 - best NO ask.
        A NO seller at 45c implies a YES buyer at 55c.
        """
        no_levels = orderbook.get("no", [])
        if not no_levels:
            return None
        no_prices = [lvl["price"] for lvl in no_levels if lvl.get("delta", 0) > 0]
        if not no_prices:
            return None
        return 100 - min(no_prices)

    # ------------------------------------------------------------------
    # Limit Ladder Execution (DEEP tier, wide spread)
    # ------------------------------------------------------------------

    async def execute_limit_ladder(
        self,
        contract: ChosenContract,
        total_contracts: int,
        execution_mode: ExecutionMode,
    ) -> list[PendingOrder]:
        """
        Split order across 2-3 price levels inside the spread on DEEP books.

        Spread <= 3c: single order (no room for ladder)
        Spread 4-6c: 2 levels — bid+2 (60%) + bid+1 (40%)
        Spread > 6c:  3 levels — midpoint (40%) + bid+1 (35%) + bid (25%)

        Every level independently satisfies min_edge_threshold.
        Total contract count is split, not increased.
        """
        orderbook = await self.get_market_orderbook(contract.ticker)
        if not orderbook:
            return []

        if contract.side == Side.YES:
            best_ask = self.best_yes_ask(orderbook)
            best_bid = self.best_yes_bid(orderbook)
        else:
            best_ask = self.best_no_ask(orderbook)
            ya = self.best_yes_ask(orderbook)
            best_bid = (100 - ya) if ya is not None else None

        if best_ask is None or best_bid is None or best_ask <= best_bid:
            return []

        spread = best_ask - best_bid
        model_p_cents = int(contract.p_model * 100)
        min_edge_cents = max(1, int(self.cfg.min_edge * 100))
        max_price = model_p_cents - min_edge_cents  # Edge floor

        # Compute ladder levels
        if spread <= 6:
            # 2-level ladder
            n1 = max(1, int(total_contracts * 0.6))
            n2 = total_contracts - n1
            levels = [
                (min(best_bid + 2, max_price), n1),
                (min(best_bid + 1, max_price), n2),
            ]
        else:
            # 3-level ladder
            n1 = max(1, int(total_contracts * 0.4))
            n2 = max(1, int(total_contracts * 0.35))
            n3 = max(1, total_contracts - n1 - n2)
            mid = (best_ask + best_bid) // 2
            levels = [
                (min(mid, max_price), n1),
                (min(best_bid + 1, max_price), n2),
                (min(best_bid, max_price), n3),
            ]

        # Deduplicate and filter invalid levels
        seen_prices: set[int] = set()
        valid_levels: list[tuple[int, int]] = []
        for price, qty in levels:
            price = max(1, min(99, price))
            if qty <= 0 or price in seen_prices:
                continue
            edge = contract.p_model - (price / 100.0)
            if edge < self.cfg.min_edge:
                continue
            seen_prices.add(price)
            valid_levels.append((price, qty))

        if not valid_levels:
            return []

        # Check resting capacity
        available_slots = self.MAX_RESTING_ORDERS - self.resting_count
        if available_slots <= 0:
            return []
        valid_levels = valid_levels[:available_slots]

        orders: list[PendingOrder] = []
        for price_cents, qty in valid_levels:
            edge = contract.p_model - (price_cents / 100.0)
            total_cost = price_cents * qty

            allowed, reason = self.risk.passes_risk_checks_for_stink_bid(edge, total_cost)
            if not allowed:
                continue

            instruction = TradeInstruction(
                ticker=contract.ticker,
                action=OrderAction.BUY,
                side=contract.side,
                contracts=qty,
                limit_price_cents=price_cents,
                model_probability=contract.p_model,
                edge=edge,
                horizon=contract.horizon,
                liquidity_tier=contract.liquidity_tier,
                metadata={"strategy": "limit_ladder", "level_price": price_cents},
            )

            result = await self._submit_order(instruction)
            if result:
                oid = result.get("order", {}).get("order_id", "unknown")
                strike = contract.metadata.get("strike") if contract.metadata else None
                ref_bid = best_bid

                pending = PendingOrder(
                    order_id=oid,
                    ticker=contract.ticker,
                    side=contract.side,
                    limit_price_cents=price_cents,
                    contracts=qty,
                    horizon=contract.horizon,
                    liquidity_tier=contract.liquidity_tier,
                    model_p_at_placement=contract.p_model,
                    edge_at_placement=edge,
                    expiry_utc=contract.close_time,
                    execution_mode=execution_mode.value,
                    strike=strike,
                    metadata={"reference_bid": ref_bid, "improvements": 0},
                )
                self._pending_orders[oid] = pending
                self.risk.notify_resting_order_placed(total_cost)
                orders.append(pending)

                logger.info(
                    "LADDER LEVEL | %s %s x%d @%dc | edge=%.4f (level %d/%d)",
                    contract.side.value, contract.ticker,
                    qty, price_cents, edge,
                    len(orders), len(valid_levels),
                )

        return orders

    # ------------------------------------------------------------------
    # Unified Execute from ChosenContract
    # ------------------------------------------------------------------

    async def execute_chosen_contract(
        self,
        contract: ChosenContract,
        num_contracts: int,
        execution_mode: ExecutionMode | None = None,
    ) -> TradeInstruction | PendingOrder | None:
        """
        Execute a trade for a contract selected by the router.

        Resting-order-first workflow:
        - EMPTY tier → stink bid (unchanged)
        - PASSIVE / MODERATE → post resting limit inside the spread
        - AGGRESSIVE → take the best ask
        - SKIP → no trade
        """
        if num_contracts <= 0:
            logger.info("No contracts to trade for %s", contract.ticker)
            return None

        tier = contract.liquidity_tier
        mode = execution_mode or ExecutionMode.AGGRESSIVE

        # Kill switch: if guard is in stink-only mode, only allow EMPTY stinks
        if self.guard.is_stink_only and tier != LiquidityTier.EMPTY:
            logger.info(
                "Guard STINK-ONLY | skipping %s (tier=%s) — degraded mode active",
                contract.ticker, tier.value,
            )
            return None

        # EMPTY tier — delegate to stink bid (already passive)
        if tier == LiquidityTier.EMPTY:
            return await self.execute_stink_bid(contract, num_contracts)

        if mode == ExecutionMode.SKIP:
            logger.info("Execution skipped (mode=SKIP) for %s", contract.ticker)
            return None

        # DEEP tier + PASSIVE + spread > 3c + multiple contracts → limit ladder
        if (
            tier == LiquidityTier.DEEP
            and mode == ExecutionMode.PASSIVE
            and num_contracts > 1
            and contract.spread > 0.03
        ):
            ladder_orders = await self.execute_limit_ladder(contract, num_contracts, mode)
            if ladder_orders:
                return ladder_orders[0]  # Return first level as primary result

        # PASSIVE or MODERATE — post resting limit order
        if mode in (ExecutionMode.PASSIVE, ExecutionMode.MODERATE):
            return await self.execute_passive_limit(contract, num_contracts, mode)

        # AGGRESSIVE — take best ask
        return await self._execute_deep_book(contract, num_contracts)

    async def _execute_deep_book(
        self,
        contract: ChosenContract,
        num_contracts: int,
    ) -> TradeInstruction | None:
        """DEEP tier: buy at best ask (original behavior)."""
        orderbook = await self.get_market_orderbook(contract.ticker)
        ask_cents: int | None = None

        if orderbook:
            if contract.side == Side.YES:
                ask_cents = self.best_yes_ask(orderbook)
            else:
                ask_cents = self.best_no_ask(orderbook)

        if ask_cents is None:
            logger.warning(
                "No live orderbook depth for %s %s — skipping trade",
                contract.ticker, contract.horizon,
            )
            return None

        # Verify edge still exists with live price
        live_implied = ask_cents / 100.0
        live_edge = contract.p_model - live_implied

        if live_edge < contract.edge * 0.5:
            logger.info(
                "Edge deteriorated for %s: expected=%.4f live=%.4f — skipping",
                contract.ticker, contract.edge, live_edge,
            )
            return None

        instruction = TradeInstruction(
            ticker=contract.ticker,
            action=OrderAction.BUY,
            side=contract.side,
            contracts=num_contracts,
            limit_price_cents=ask_cents,
            model_probability=contract.p_model,
            edge=live_edge,
            horizon=contract.horizon,
            liquidity_tier=LiquidityTier.DEEP,
            metadata={
                "router_edge": contract.edge,
                "live_edge": live_edge,
                "orderbook_time": datetime.now(timezone.utc).isoformat(),
            },
        )

        logger.info(
            "TRADE SIGNAL | %s %s %s x%d @%dc | edge=%.4f p_model=%.4f horizon=%s",
            instruction.action.value.upper(), instruction.side.value.upper(),
            contract.ticker, num_contracts, ask_cents, live_edge,
            contract.p_model, contract.horizon,
        )

        result = await self._submit_order(instruction)
        if result:
            self.risk.notify_position_opened()
            return instruction
        # Submission failed — return None so caller does NOT register phantom position
        return None

    # ------------------------------------------------------------------
    # Legacy: Evaluate & Execute (single contract, daily only)
    # ------------------------------------------------------------------

    async def evaluate_and_execute(
        self,
        ticker: str,
        model_probability: float,
    ) -> TradeInstruction | None:
        """
        Legacy end-to-end evaluation for a single contract.
        Kept for backward compatibility.
        """
        orderbook = await self.get_market_orderbook(ticker)
        if not orderbook:
            logger.warning("Empty orderbook for %s — skipping", ticker)
            return None

        ask_cents = self.best_yes_ask(orderbook)
        if ask_cents is None:
            logger.warning("No YES ask available for %s — skipping", ticker)
            return None

        yes_edge = self.calculate_edge(model_probability, ask_cents, Side.YES)

        if yes_edge >= self.cfg.min_edge:
            side = Side.YES
            edge = yes_edge
            limit_price = ask_cents
        else:
            no_ask_levels = orderbook.get("no", [])
            no_prices = [lvl["price"] for lvl in no_ask_levels if lvl.get("delta", 0) > 0]
            if not no_prices:
                logger.info("No edge on YES (%.4f) and no NO liquidity for %s", yes_edge, ticker)
                return None
            no_ask = min(no_prices)
            no_edge = self.calculate_edge(1.0 - model_probability, no_ask, Side.YES)
            if no_edge < self.cfg.min_edge:
                logger.info("No actionable edge for %s (YES=%.4f, NO=%.4f)", ticker, yes_edge, no_edge)
                return None
            side = Side.NO
            edge = no_edge
            limit_price = no_ask

        allowed, reason = self.risk.passes_risk_checks(edge)
        if not allowed:
            logger.info("Risk check blocked trade on %s: %s", ticker, reason)
            return None

        contracts = self.risk.compute_kelly_contracts(
            model_prob=model_probability if side == Side.YES else 1.0 - model_probability,
            ask_price_cents=limit_price,
        )
        if contracts == 0:
            logger.info("Kelly sizing returned 0 contracts for %s — no trade", ticker)
            return None

        instruction = TradeInstruction(
            ticker=ticker,
            action=OrderAction.BUY,
            side=side,
            contracts=contracts,
            limit_price_cents=limit_price,
            model_probability=model_probability,
            edge=edge,
            metadata={"orderbook_snapshot_time": datetime.now(timezone.utc).isoformat()},
        )

        logger.info(
            "TRADE SIGNAL | %s %s %s x%d @%dc | edge=%.4f p_model=%.4f",
            instruction.action.value.upper(), instruction.side.value.upper(),
            ticker, contracts, limit_price, edge, model_probability,
        )

        await self._submit_order(instruction)
        return instruction

    # ------------------------------------------------------------------
    # Order Submission
    # ------------------------------------------------------------------

    async def _submit_order(self, instr: TradeInstruction) -> dict[str, Any] | None:
        """
        POST the limit order to Kalshi.

        Kalshi v2 order payload:
        {
            "ticker": "...",
            "action": "buy",
            "side": "yes",
            "type": "limit",
            "count": 5,
            "yes_price": 65   (or no_price)
        }
        """
        self._ensure_auth()
        url = f"{self.cfg.api_base}/portfolio/orders"
        payload: dict[str, Any] = {
            "ticker": instr.ticker,
            "action": instr.action.value,
            "side": instr.side.value,
            "type": "limit",
            "count": instr.contracts,
        }
        if instr.side == Side.YES:
            payload["yes_price"] = instr.limit_price_cents
        else:
            payload["no_price"] = instr.limit_price_cents

        assert self._session is not None
        try:
            auth_headers = self._sign_request("POST", url)
            async with self._session.post(url, json=payload, headers=auth_headers,
                                          timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 201:
                    data = await resp.json()
                    order_id = data.get("order", {}).get("order_id", "unknown")
                    logger.info("Order submitted | id=%s ticker=%s horizon=%s tier=%s",
                               order_id, instr.ticker, instr.horizon,
                               instr.liquidity_tier.value)
                    # Caller is responsible for notify_position_opened() (not stink bids)
                    self.guard.record_submission()
                    return data
                else:
                    body = await resp.text()
                    logger.error("Order submission failed [%d]: %s", resp.status, body)
                    return None
        except Exception:
            logger.exception("Order submission error for %s", instr.ticker)
            return None

    # ------------------------------------------------------------------
    # Resting-Order Reevaluation (called after signal generation)
    # ------------------------------------------------------------------

    def _reevaluate_resting_order(
        self,
        pending: PendingOrder,
        signal_engine: Any,
        state: Any,
        oracle: Any,
        now: datetime,
    ) -> str:
        """
        Reevaluate a resting order against current market conditions.

        Returns action:
        - "hold"     — keep resting, edge is still valid
        - "cancel"   — edge deteriorated below threshold
        - "escalate" — time is short but edge is strong, take aggressively
        """
        current_price = state.mid_price if state.mid_price > 0 else (
            state.prices[-1] if state.prices else 0.0
        )
        if current_price <= 0:
            return "hold"

        model_p = pending.model_p_at_placement  # fallback

        if pending.strike is not None and pending.expiry_utc is not None:
            minutes_remaining = max(
                0.0, (pending.expiry_utc - now).total_seconds() / 60.0,
            )
            try:
                strike_eval = signal_engine.evaluate_contract(
                    current_price, pending.strike, minutes_remaining, state, oracle,
                )
                raw_p = strike_eval["p_above"]
                model_p = raw_p if pending.side == Side.YES else 1.0 - raw_p
            except Exception:
                pass

        current_edge = model_p - (pending.limit_price_cents / 100.0)

        secs_remaining = 900.0
        if pending.expiry_utc:
            secs_remaining = max(0, (pending.expiry_utc - now).total_seconds())

        min_edge = self.cfg.min_edge

        # Cancel if edge dropped below minimum threshold
        if current_edge < min_edge:
            logger.info(
                "REEVAL cancel | %s edge deteriorated: %.4f (was %.4f, min %.4f)",
                pending.ticker, current_edge, pending.edge_at_placement, min_edge,
            )
            return "cancel"

        # THIN_MARGINAL: never escalate to aggressive
        if pending.liquidity_tier == LiquidityTier.THIN_MARGINAL:
            return "hold"

        # Stink bids: no escalation pipeline
        if pending.execution_mode not in ("passive", "moderate"):
            return "hold"

        # Track improvements already made (max 2 per order lifetime)
        improvements = pending.metadata.get("improvements", 0)

        # -------------------------------------------------------------------
        # Graduated escalation pipeline (4 phases by time remaining)
        # -------------------------------------------------------------------

        # Phase 4: <45s — take the ask if edge justifies it
        if secs_remaining < 45:
            if current_edge >= min_edge * 0.75:
                logger.info(
                    "REEVAL escalate | %s %.0fs left, edge=%.4f → AGGRESSIVE",
                    pending.ticker, secs_remaining, current_edge,
                )
                return "escalate"
            return "cancel"

        # Phase 3: 45-90s — improve to midpoint (if budget allows)
        if secs_remaining < 90 and improvements < 2:
            return "improve_to_mid"

        # Phase 2: 90-180s — improve by +1 tick (if budget allows)
        if secs_remaining < 180 and improvements < 2:
            new_price = pending.limit_price_cents + 1
            new_edge = model_p - (new_price / 100.0)
            if new_edge >= min_edge:
                return "improve_1tick"

        # Phase 1: >180s — hold
        return "hold"

    async def reevaluate_resting_orders(
        self,
        signal_engine: Any,
        state: Any,
        oracle: Any,
        router: Any | None = None,
    ) -> tuple[int, int, int]:
        """
        Reevaluate all resting orders against current model/market state.

        Cancels orders where edge has deteriorated.
        Escalates orders where time is short but edge justifies aggressive fill.

        Returns (escalated_count, cancelled_count, still_resting_count).
        """
        now = datetime.now(timezone.utc)
        escalated = 0
        cancelled = 0
        still_resting = 0

        for order_id, pending in list(self._pending_orders.items()):
            if pending.status != "resting":
                continue

            action = self._reevaluate_resting_order(
                pending, signal_engine, state, oracle, now,
            )

            if action == "cancel":
                await self.cancel_order(order_id)
                pending.status = "cancelled"
                cost = pending.limit_price_cents * pending.contracts
                self.risk.notify_resting_order_resolved(cost, was_filled=False)
                cancelled += 1
                if router is not None:
                    has_other = any(
                        o.ticker == pending.ticker
                        and o.status == "resting"
                        and o.order_id != order_id
                        for o in self._pending_orders.values()
                    )
                    if not has_other:
                        router.unregister_position(pending.ticker, pending.contracts)

            elif action == "escalate":
                result = await self._escalate_to_aggressive(order_id, pending, router)
                if result:
                    escalated += 1
                else:
                    cancelled += 1

            elif action in ("improve_1tick", "improve_to_mid"):
                new_price = await self._compute_improvement_price(
                    pending, action,
                )
                if new_price is not None and new_price != pending.limit_price_cents:
                    result = await self._improve_resting_order(
                        order_id, pending, new_price,
                    )
                    if result:
                        still_resting += 1  # Repriced, still resting
                    else:
                        cancelled += 1
                else:
                    still_resting += 1  # No valid improvement price, hold

            else:
                still_resting += 1

        return escalated, cancelled, still_resting

    async def _should_repeg(self, pending: PendingOrder) -> int | None:
        """
        Bid pegging for THIN_TRADEABLE / MODERATE orders.

        If the best bid has moved UP by >=2c since order placement,
        reprice to new_bid + 1. Never chase downward (adverse selection).
        Only applies if improvement budget not exhausted (max 2).

        Returns new price in cents, or None if no repeg needed.
        """
        # Only peg THIN_TRADEABLE moderate orders
        if pending.liquidity_tier != LiquidityTier.THIN_TRADEABLE:
            return None
        if pending.execution_mode != "moderate":
            return None
        if pending.metadata.get("improvements", 0) >= 2:
            return None

        ref_bid = pending.metadata.get("reference_bid")
        if ref_bid is None:
            return None

        orderbook = await self.get_market_orderbook(pending.ticker)
        if not orderbook:
            return None

        if pending.side == Side.YES:
            current_bid = self.best_yes_bid(orderbook)
        else:
            ya = self.best_yes_ask(orderbook)
            current_bid = (100 - ya) if ya is not None else None

        if current_bid is None:
            return None

        # Bid moved up by >= 2c — repeg
        if current_bid >= ref_bid + 2:
            new_price = current_bid + 1
            new_price = max(1, min(99, new_price))
            # Edge floor
            edge = pending.model_p_at_placement - (new_price / 100.0)
            if edge >= self.cfg.min_edge:
                logger.info(
                    "REPEG | %s bid moved %dc → %dc, repricing %dc → %dc",
                    pending.ticker, ref_bid, current_bid,
                    pending.limit_price_cents, new_price,
                )
                return new_price

        return None

    async def _compute_improvement_price(
        self,
        pending: PendingOrder,
        action: str,
    ) -> int | None:
        """Compute the new price for a graduated price improvement."""
        min_edge = self.cfg.min_edge
        model_p_cents = int(pending.model_p_at_placement * 100)
        min_edge_cents = max(1, int(min_edge * 100))
        max_price = model_p_cents - min_edge_cents

        if action == "improve_1tick":
            new_price = pending.limit_price_cents + 1
        elif action == "improve_to_mid":
            orderbook = await self.get_market_orderbook(pending.ticker)
            if not orderbook:
                return None
            if pending.side == Side.YES:
                best_ask = self.best_yes_ask(orderbook)
                best_bid = self.best_yes_bid(orderbook)
            else:
                best_ask = self.best_no_ask(orderbook)
                yes_ask = self.best_yes_ask(orderbook)
                best_bid = (100 - yes_ask) if yes_ask is not None else None
            if best_ask is None or best_bid is None:
                return None
            new_price = (best_ask + best_bid) // 2
        else:
            return None

        new_price = max(1, min(99, new_price))
        # Edge floor: never exceed max_price
        new_price = min(new_price, max_price)
        # Must actually improve (higher price = more aggressive)
        if new_price <= pending.limit_price_cents:
            return None
        return new_price

    async def _improve_resting_order(
        self,
        order_id: str,
        pending: PendingOrder,
        new_price_cents: int,
    ) -> bool:
        """
        Cancel a resting order and repost at an improved price.
        Atomic from risk perspective: free old cost, register new cost.
        Returns True if the new order was placed successfully.
        """
        old_cost = pending.limit_price_cents * pending.contracts
        await self.cancel_order(order_id)
        self.risk.notify_resting_order_resolved(old_cost, was_filled=False)

        new_edge = pending.model_p_at_placement - (new_price_cents / 100.0)
        instruction = TradeInstruction(
            ticker=pending.ticker,
            action=OrderAction.BUY,
            side=pending.side,
            contracts=pending.contracts,
            limit_price_cents=new_price_cents,
            model_probability=pending.model_p_at_placement,
            edge=new_edge,
            horizon=pending.horizon,
            liquidity_tier=pending.liquidity_tier,
            metadata={
                "strategy": f"improved_from_{pending.limit_price_cents}c",
            },
        )

        result = await self._submit_order(instruction)
        if result:
            new_oid = result.get("order", {}).get("order_id", "unknown")
            new_pending = PendingOrder(
                order_id=new_oid,
                ticker=pending.ticker,
                side=pending.side,
                limit_price_cents=new_price_cents,
                contracts=pending.contracts,
                horizon=pending.horizon,
                liquidity_tier=pending.liquidity_tier,
                model_p_at_placement=pending.model_p_at_placement,
                edge_at_placement=new_edge,
                expiry_utc=pending.expiry_utc,
                execution_mode=pending.execution_mode,
                strike=pending.strike,
                cycles_alive=pending.cycles_alive,  # Preserve age
                metadata={
                    **pending.metadata,
                    "improvements": pending.metadata.get("improvements", 0) + 1,
                    "reference_bid": pending.metadata.get("reference_bid"),
                },
            )
            del self._pending_orders[order_id]
            self._pending_orders[new_oid] = new_pending

            new_cost = new_price_cents * pending.contracts
            self.risk.notify_resting_order_placed(new_cost)
            logger.info(
                "IMPROVE | %s %dc → %dc (improvement #%d) | edge=%.4f",
                pending.ticker, pending.limit_price_cents, new_price_cents,
                new_pending.metadata.get("improvements", 1), new_edge,
            )
            return True

        # Submission failed — order cancelled but not replaced
        logger.warning("Improve failed for %s — order lost", pending.ticker)
        pending.status = "cancelled"
        del self._pending_orders[order_id]
        return False

    async def _escalate_to_aggressive(
        self,
        order_id: str,
        pending: PendingOrder,
        router: Any | None = None,
    ) -> bool:
        """
        Cancel a passive resting order and immediately take the best ask.
        Returns True if the aggressive order was placed successfully.
        """
        # 1. Cancel the passive order
        await self.cancel_order(order_id)
        pending.status = "cancelled"
        cost = pending.limit_price_cents * pending.contracts
        self.risk.notify_resting_order_resolved(cost, was_filled=False)

        # 2. Fetch current orderbook
        orderbook = await self.get_market_orderbook(pending.ticker)
        if not orderbook:
            logger.warning("Escalation failed: no orderbook for %s", pending.ticker)
            if router:
                router.unregister_position(pending.ticker, pending.contracts)
            return False

        # 3. Find best ask
        if pending.side == Side.YES:
            ask_cents = self.best_yes_ask(orderbook)
        else:
            ask_cents = self.best_no_ask(orderbook)

        if ask_cents is None:
            logger.warning("Escalation failed: no ask for %s", pending.ticker)
            if router:
                router.unregister_position(pending.ticker, pending.contracts)
            return False

        # 4. Verify edge at take price
        live_edge = pending.model_p_at_placement - (ask_cents / 100.0)
        if live_edge < self.cfg.min_edge * 0.75:
            logger.info(
                "Escalation aborted | %s: edge at take price (%.4f) insufficient",
                pending.ticker, live_edge,
            )
            if router:
                router.unregister_position(pending.ticker, pending.contracts)
            return False

        # 5. Risk check for aggressive fill
        allowed, reason = self.risk.passes_risk_checks(live_edge)
        if not allowed:
            logger.info("Escalation blocked by risk: %s", reason)
            if router:
                router.unregister_position(pending.ticker, pending.contracts)
            return False

        # 6. Submit aggressive order
        instruction = TradeInstruction(
            ticker=pending.ticker,
            action=OrderAction.BUY,
            side=pending.side,
            contracts=pending.contracts,
            limit_price_cents=ask_cents,
            model_probability=pending.model_p_at_placement,
            edge=live_edge,
            horizon=pending.horizon,
            liquidity_tier=pending.liquidity_tier,
            metadata={
                "strategy": "escalated_aggressive",
                "passive_price": pending.limit_price_cents,
                "cycles_passive": pending.cycles_alive,
            },
        )

        logger.info(
            "ESCALATE | %s %s x%d @%dc (was passive @%dc for %d cycles) | edge=%.4f",
            instruction.side.value, pending.ticker,
            pending.contracts, ask_cents,
            pending.limit_price_cents, pending.cycles_alive, live_edge,
        )

        result = await self._submit_order(instruction)
        if result:
            self.risk.notify_position_opened()
            # Router registration already exists from the passive order — keep it
            return True

        # Submission failed — clean up router
        if router:
            router.unregister_position(pending.ticker, pending.contracts)
        return False
