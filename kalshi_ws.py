"""
Kalshi WebSocket Client — Real-time orderbook streaming.

Maintains local orderbooks for subscribed markets via the Kalshi v2
WebSocket API, providing always-fresh bid/ask data to the router and
eliminating the stale-listing-price problem.

Data flow
---------
1. Authenticate WebSocket handshake with RSA-PSS signing.
2. Subscribe to ``orderbook_delta`` channel for active market tickers.
3. Receive ``orderbook_snapshot`` → seed local book.
4. Receive ``orderbook_delta`` → apply incremental updates.
5. On demand: return local book state for routing and execution.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import Any

import aiohttp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from kalshi_client import KalshiConfig, KALSHI_API_BASE_DEMO

logger = logging.getLogger(__name__)

STALE_THRESHOLD_S = 30.0


# ---------------------------------------------------------------------------
# Local Orderbook (Kalshi)
# ---------------------------------------------------------------------------

class KalshiLocalOrderbook:
    """
    Maintains a price-level orderbook for a single Kalshi contract.

    Prices are in cents (1-99).  Quantities are number of contracts.
    """

    def __init__(self) -> None:
        self.yes_levels: dict[int, int] = {}  # price_cents -> qty
        self.no_levels: dict[int, int] = {}
        self.last_update: float = 0.0

    def apply_snapshot(self, yes_list: list, no_list: list) -> None:
        """Replace full book from [[price, qty], ...] arrays."""
        self.yes_levels.clear()
        self.no_levels.clear()
        for price, qty in yes_list:
            p, q = int(price), int(qty)
            if q > 0:
                self.yes_levels[p] = q
        for price, qty in no_list:
            p, q = int(price), int(qty)
            if q > 0:
                self.no_levels[p] = q
        self.last_update = time.monotonic()

    def apply_delta(self, yes_list: list, no_list: list) -> None:
        """Incremental update; remove levels where qty=0."""
        for price, qty in yes_list:
            p, q = int(price), int(qty)
            if q == 0:
                self.yes_levels.pop(p, None)
            else:
                self.yes_levels[p] = q
        for price, qty in no_list:
            p, q = int(price), int(qty)
            if q == 0:
                self.no_levels.pop(p, None)
            else:
                self.no_levels[p] = q
        self.last_update = time.monotonic()

    @property
    def best_yes_ask(self) -> int | None:
        """Lowest YES price with qty > 0."""
        return min(self.yes_levels) if self.yes_levels else None

    @property
    def best_yes_bid(self) -> int | None:
        """Highest YES bid = 100 - best NO ask (lowest NO price with qty)."""
        if not self.no_levels:
            return None
        return 100 - min(self.no_levels)

    @property
    def best_no_ask(self) -> int | None:
        """Lowest NO price with qty > 0."""
        return min(self.no_levels) if self.no_levels else None

    @property
    def best_no_bid(self) -> int | None:
        """Highest NO price with qty > 0."""
        return max(self.no_levels) if self.no_levels else None

    @property
    def yes_spread_cents(self) -> int:
        ask, bid = self.best_yes_ask, self.best_yes_bid
        if ask is None or bid is None:
            return 99  # Unknown spread — treat as wide, not zero
        return max(0, ask - bid)

    @property
    def has_depth(self) -> bool:
        """True if there is at least one resting level on either side."""
        return bool(self.yes_levels) or bool(self.no_levels)

    @property
    def total_depth_contracts(self) -> int:
        """Total resting contracts across all levels on both sides."""
        return sum(self.yes_levels.values()) + sum(self.no_levels.values())

    def classify_liquidity(
        self,
        min_deep_contracts: int = 50,
        min_deep_levels: int = 4,
        max_deep_spread: int = 5,
    ) -> str:
        """
        Classify the orderbook into a liquidity tier string.

        Thresholds are passed in by the caller (router) so they can be
        adjusted dynamically based on time-of-day, volatility, etc.

        Returns one of: "deep", "thin_tradeable", "thin_marginal", "empty".
        """
        if not self.has_depth:
            return "empty"
        total = self.total_depth_contracts
        num_levels = len(self.yes_levels) + len(self.no_levels)
        spread = self.yes_spread_cents
        if total >= min_deep_contracts and num_levels >= min_deep_levels and spread <= max_deep_spread:
            return "deep"
        # THIN_TRADEABLE: relaxed but still actionable
        if total >= 10 and num_levels >= 2 and spread <= 10:
            return "thin_tradeable"
        return "thin_marginal"

    @property
    def liquidity_tier(self) -> str:
        """Default classification with base thresholds (backwards compat)."""
        return self.classify_liquidity()

    def to_rest_format(self) -> dict[str, Any]:
        """
        Return data formatted like the Kalshi REST orderbook response,
        so existing code in KalshiClient can consume it unchanged.

        REST format: {"yes": [{"price": 55, "delta": 10}, ...],
                      "no":  [{"price": 45, "delta": 5}, ...]}
        """
        yes = [{"price": p, "delta": q} for p, q in sorted(self.yes_levels.items())]
        no = [{"price": p, "delta": q} for p, q in sorted(self.no_levels.items())]
        return {"yes": yes, "no": no}


# ---------------------------------------------------------------------------
# Kalshi WebSocket Client
# ---------------------------------------------------------------------------

class KalshiWebSocket:
    """
    Async WebSocket client that subscribes to Kalshi orderbook_delta
    channel and maintains local orderbooks for active markets.
    """

    def __init__(self, config: KalshiConfig) -> None:
        self.cfg = config
        self._books: dict[str, KalshiLocalOrderbook] = {}
        self._subscribed_tickers: set[str] = set()
        self._ticker_to_sid: dict[str, int] = {}   # ticker -> subscription id
        self._sid_to_ticker: dict[int, str] = {}    # reverse lookup
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._msg_id: int = 0
        self._running = False
        self._task: asyncio.Task | None = None
        self._private_key = None
        self._connected = asyncio.Event()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _load_private_key(self) -> None:
        with open(self.cfg.private_key_path, "rb") as f:
            self._private_key = serialization.load_pem_private_key(
                f.read(), password=None,
            )

    def _sign_ws_handshake(self) -> dict[str, str]:
        """
        Sign the WebSocket handshake.

        Kalshi WS signing: message = timestamp_ms + "GET" + "/trade-api/ws/v2"
        Uses PSS.DIGEST_LENGTH for salt (different from REST which uses MAX_LENGTH).
        """
        timestamp_ms = str(int(time.time() * 1000))
        msg = (timestamp_ms + "GET" + "/trade-api/ws/v2").encode()
        sig_bytes = self._private_key.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256().digest_size,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.cfg.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig_bytes).decode(),
        }

    # ------------------------------------------------------------------
    # WebSocket URL
    # ------------------------------------------------------------------

    @property
    def _ws_url(self) -> str:
        if "demo" in self.cfg.api_base:
            return "wss://demo-api.kalshi.co/trade-api/ws/v2"
        return "wss://api.elections.kalshi.com/trade-api/ws/v2"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._load_private_key()
        self._session = aiohttp.ClientSession()
        self._running = True
        self._task = asyncio.create_task(
            self._reconnect_loop(), name="kalshi_ws",
        )
        logger.info("KalshiWebSocket starting | url=%s", self._ws_url)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session:
            await self._session.close()
        logger.info("KalshiWebSocket stopped")

    # ------------------------------------------------------------------
    # Connection loop with reconnection
    # ------------------------------------------------------------------

    async def _reconnect_loop(self) -> None:
        backoff = 1.0
        while self._running:
            try:
                auth_headers = self._sign_ws_handshake()
                assert self._session is not None
                self._ws = await self._session.ws_connect(
                    self._ws_url,
                    headers=auth_headers,
                    heartbeat=30,
                    timeout=aiohttp.ClientTimeout(total=15),
                )
                logger.info("Kalshi WS connected")
                self._connected.set()
                backoff = 1.0

                # Clear sid mappings — server assigns new ones per connection
                self._ticker_to_sid.clear()
                self._sid_to_ticker.clear()

                # Re-subscribe to any previously tracked tickers
                if self._subscribed_tickers:
                    await self._send_subscribe(list(self._subscribed_tickers))

                await self._message_loop()

            except asyncio.CancelledError:
                return
            except Exception:
                self._connected.clear()
                logger.exception(
                    "Kalshi WS error — reconnecting in %.0f s", backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _message_loop(self) -> None:
        assert self._ws is not None
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = msg.json()
                self._handle_message(data)
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                logger.warning("Kalshi WS closed/error — will reconnect")
                self._connected.clear()
                break

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _handle_message(self, data: dict[str, Any]) -> None:
        msg_type = data.get("type", "")

        if msg_type == "orderbook_snapshot":
            ticker = data.get("msg", {}).get("market_ticker", "")
            if not ticker:
                return
            book = self._books.setdefault(ticker, KalshiLocalOrderbook())
            yes_list = data["msg"].get("yes", [])
            no_list = data["msg"].get("no", [])
            book.apply_snapshot(yes_list, no_list)
            logger.debug(
                "Kalshi book snapshot | %s | yes_levels=%d no_levels=%d",
                ticker, len(book.yes_levels), len(book.no_levels),
            )

        elif msg_type == "orderbook_delta":
            ticker = data.get("msg", {}).get("market_ticker", "")
            if not ticker:
                return
            book = self._books.get(ticker)
            if book is None:
                book = KalshiLocalOrderbook()
                self._books[ticker] = book
            yes_list = data["msg"].get("yes", [])
            no_list = data["msg"].get("no", [])
            book.apply_delta(yes_list, no_list)

        elif msg_type == "subscribed":
            sid = data.get("sid")
            msg_body = data.get("msg", {})
            ticker = msg_body.get("market_ticker", "")
            if sid is not None and ticker:
                self._ticker_to_sid[ticker] = sid
                self._sid_to_ticker[sid] = ticker
                logger.debug("Kalshi WS subscription confirmed | sid=%d ticker=%s", sid, ticker)

        elif msg_type == "unsubscribed":
            sid = data.get("sid")
            if sid is not None:
                ticker = self._sid_to_ticker.pop(sid, None)
                if ticker:
                    self._ticker_to_sid.pop(ticker, None)
                logger.debug("Kalshi WS unsubscribed | sid=%d ticker=%s", sid, ticker)

        elif msg_type == "error":
            logger.error("Kalshi WS error message: %s", data)

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    async def _send_subscribe(self, tickers: list[str]) -> None:
        if not self._ws or self._ws.closed:
            return
        self._msg_id += 1
        cmd = {
            "id": self._msg_id,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_tickers": tickers,
            },
        }
        await self._ws.send_json(cmd)
        logger.info("Kalshi WS subscribed to %d tickers", len(tickers))

    async def _send_unsubscribe(self, tickers: list[str]) -> None:
        if not self._ws or self._ws.closed:
            return
        sids = [self._ticker_to_sid[t] for t in tickers if t in self._ticker_to_sid]
        if not sids:
            logger.debug("Kalshi WS unsubscribe skipped — no sids for %d tickers", len(tickers))
            return
        self._msg_id += 1
        cmd = {
            "id": self._msg_id,
            "cmd": "unsubscribe",
            "params": {
                "sids": sids,
            },
        }
        await self._ws.send_json(cmd)
        logger.info("Kalshi WS unsubscribed sids=%s (%d tickers)", sids, len(tickers))

    async def subscribe_markets(self, tickers: list[str]) -> None:
        """Subscribe to orderbook updates for the given tickers."""
        new = [t for t in tickers if t not in self._subscribed_tickers]
        if not new:
            return
        self._subscribed_tickers.update(new)
        if self._connected.is_set():
            await self._send_subscribe(new)

    async def unsubscribe_markets(self, tickers: list[str]) -> None:
        """Unsubscribe from tickers and discard their books."""
        old = [t for t in tickers if t in self._subscribed_tickers]
        if not old:
            return
        self._subscribed_tickers -= set(old)
        if self._connected.is_set():
            await self._send_unsubscribe(old)
        for t in old:
            self._books.pop(t, None)
            sid = self._ticker_to_sid.pop(t, None)
            if sid is not None:
                self._sid_to_ticker.pop(sid, None)

    async def update_subscriptions(self, active_tickers: list[str]) -> None:
        """
        Diff current subscriptions vs active_tickers.
        Subscribe to new ones, unsubscribe from expired ones.
        """
        active_set = set(active_tickers)
        to_add = active_set - self._subscribed_tickers
        to_remove = self._subscribed_tickers - active_set
        if to_remove:
            await self.unsubscribe_markets(list(to_remove))
        if to_add:
            await self.subscribe_markets(list(to_add))

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_orderbook(self, ticker: str) -> KalshiLocalOrderbook | None:
        """Return the local book for a ticker, or None if not available."""
        book = self._books.get(ticker)
        if book is None:
            return None
        if not self.is_book_fresh(ticker):
            return None
        return book

    def get_best_prices(self, ticker: str) -> dict[str, Any] | None:
        """
        Return best bid/ask in cents for both sides, or None if no data.
        """
        book = self.get_orderbook(ticker)
        if book is None:
            return None
        return {
            "yes_ask_cents": book.best_yes_ask,
            "no_ask_cents": book.best_no_ask,
            "yes_bid_cents": book.best_yes_bid,
            "no_bid_cents": book.best_no_bid,
            "has_depth": book.has_depth,
            "liquidity_tier": book.liquidity_tier,
            "total_depth": book.total_depth_contracts,
            "yes_spread_cents": book.yes_spread_cents,
        }

    def is_book_fresh(self, ticker: str, max_age_seconds: float = STALE_THRESHOLD_S) -> bool:
        """Check if book data is recent enough."""
        book = self._books.get(ticker)
        if book is None or book.last_update == 0.0:
            return False
        return (time.monotonic() - book.last_update) < max_age_seconds

    def get_orderbook_rest_format(self, ticker: str) -> dict[str, Any] | None:
        """
        Return the local book in REST-compatible format, or None.
        This allows KalshiClient to use WS data transparently.
        """
        book = self.get_orderbook(ticker)
        if book is None:
            return None
        return book.to_rest_format()
