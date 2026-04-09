"""
Coinbase Microstructure Oracle.

Replaces the previous Binance US oracle with Coinbase Exchange for
dramatically better liquidity, tighter spreads, and more meaningful
microstructure signals (OBI / TFI).

Evidence (2026-04-03):
    Binance US BTC-USD:  ~0 BTC/min, $16 spread, ~1 trade/30s
    Coinbase  BTC-USD:   ~2.5 BTC/min, $0.01 spread, ~4 trades/s

Maintains a real-time L2 order book via WebSocket, computes OBI and TFI,
and provides 1m kline data from REST bootstrap + trade-stream construction.

Data flow
---------
1. Bootstrap historical 1m klines via REST ``/products/BTC-USD/candles``
2. Connect single WebSocket to ``wss://ws-feed.exchange.coinbase.com``
3. ``level2`` channel: snapshot → seed local book, ``l2update`` → apply diffs
4. ``matches`` channel: accumulate trades for TFI + build in-progress klines
5. On demand: compute OBI, TFI, apply oracle filter; serve kline data.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"
COINBASE_REST_BASE = "https://api.exchange.coinbase.com"

DEFAULT_PRODUCT = "BTC-USD"
TFI_WINDOW_SECONDS = 300.0  # 5 min rolling window for trade flow

# Kline buffer sizes — 1m only for pure 15m scalper
KLINE_BUFFER_SIZES: dict[str, int] = {
    "1m": 100,
}


# ---------------------------------------------------------------------------
# KlineBar
# ---------------------------------------------------------------------------

@dataclass
class KlineBar:
    """Represents a single kline (candlestick) bar."""
    open_time: int       # milliseconds since epoch
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool


# ---------------------------------------------------------------------------
# Local Order Book
# ---------------------------------------------------------------------------

@dataclass
class OrderBookLevel:
    price: float
    qty: float


@dataclass
class LocalOrderBook:
    """Sorted L2 book maintained from Coinbase WebSocket snapshot + updates."""

    bids: dict[float, float] = field(default_factory=dict)  # price -> qty
    asks: dict[float, float] = field(default_factory=dict)

    def apply_snapshot(self, bids: list, asks: list) -> None:
        """Seed from Coinbase level2 snapshot message."""
        self.bids = {float(p): float(s) for p, s in bids}
        self.asks = {float(p): float(s) for p, s in asks}
        logger.info(
            "Order book snapshot applied | bids=%d asks=%d",
            len(self.bids), len(self.asks),
        )

    def apply_update(self, changes: list) -> None:
        """
        Apply Coinbase l2update changes.

        Each change is ``["buy"/"sell", "price", "size"]``.
        Size of ``"0"`` means remove the level.
        """
        for side, price_s, size_s in changes:
            p, s = float(price_s), float(size_s)
            book_side = self.bids if side == "buy" else self.asks
            if s == 0.0:
                book_side.pop(p, None)
            else:
                book_side[p] = s

    @property
    def best_bid(self) -> float:
        return max(self.bids) if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return min(self.asks) if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        bb, ba = self.best_bid, self.best_ask
        if bb == 0.0 or ba == 0.0:
            return 0.0
        return (bb + ba) / 2.0

    @property
    def spread_bps(self) -> float:
        mid = self.mid_price
        if mid == 0.0:
            return 0.0
        return ((self.best_ask - self.best_bid) / mid) * 10_000


# ---------------------------------------------------------------------------
# Coinbase Oracle
# ---------------------------------------------------------------------------

class CoinbaseOracle:
    """
    Async WebSocket client for Coinbase Exchange.

    Maintains a local L2 order book, rolling trade buffer, and 1m kline bars.
    Drop-in replacement for the previous BinanceOracle — same public interface.
    """

    def __init__(
        self,
        product_id: str = DEFAULT_PRODUCT,
        tfi_window: float = TFI_WINDOW_SECONDS,
        *,
        # Accept and ignore `symbol` kwarg for backward compat with main.py
        symbol: str | None = None,
    ) -> None:
        self.product_id = product_id
        self.tfi_window = tfi_window

        self.book = LocalOrderBook()
        # Rolling buffer of (monotonic_ts, signed_qty) for TFI
        self._trades: deque[tuple[float, float]] = deque()
        self._running = False
        self._session: aiohttp.ClientSession | None = None
        self._tasks: list[asyncio.Task] = []

        # Kline buffers — keyed by interval, each a deque of KlineBar
        self._kline_buffers: dict[str, deque[KlineBar]] = {
            interval: deque(maxlen=maxlen)
            for interval, maxlen in KLINE_BUFFER_SIZES.items()
        }
        # Current in-progress (unclosed) kline for each interval
        self._current_klines: dict[str, KlineBar | None] = {
            interval: None for interval in KLINE_BUFFER_SIZES
        }
        self._klines_ready = asyncio.Event()
        self._book_ready = asyncio.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Bootstrap klines via REST, then launch the WebSocket listener."""
        self._session = aiohttp.ClientSession()
        self._running = True

        await self._bootstrap_klines()

        # Single WS connection handles both order book (level2) and trades (matches)
        self._tasks = [
            asyncio.create_task(self._ws_stream(), name="coinbase_ws"),
        ]
        logger.info("CoinbaseOracle started for %s", self.product_id)

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        if self._session:
            await self._session.close()
        logger.info("CoinbaseOracle stopped")

    # ------------------------------------------------------------------
    # Kline bootstrap (REST backfill)
    # ------------------------------------------------------------------

    async def _bootstrap_klines(self) -> None:
        """
        Fetch historical 1m candles from Coinbase REST API.
        WebSocket ``matches`` stream only sends live trades, not history.
        """
        url = f"{COINBASE_REST_BASE}/products/{self.product_id}/candles"
        assert self._session is not None

        for interval, buffer_size in KLINE_BUFFER_SIZES.items():
            if interval != "1m":
                continue  # only 1m supported

            params = {"granularity": "60"}  # 60 seconds = 1m
            try:
                async with self._session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    resp.raise_for_status()
                    raw = await resp.json()
            except Exception:
                logger.exception("Failed to bootstrap klines for %s", interval)
                continue

            if not raw:
                continue

            # Coinbase returns candles in REVERSE chronological order — flip.
            # Format: [timestamp_s, low, high, open, close, volume]
            raw = list(reversed(raw))

            buf = self._kline_buffers[interval]
            # All but last are closed; last is in-progress.
            for k in raw[:-1]:
                buf.append(KlineBar(
                    open_time=int(k[0]) * 1000,   # seconds → ms
                    open=float(k[3]),
                    high=float(k[2]),
                    low=float(k[1]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    is_closed=True,
                ))

            last = raw[-1]
            self._current_klines[interval] = KlineBar(
                open_time=int(last[0]) * 1000,
                open=float(last[3]),
                high=float(last[2]),
                low=float(last[1]),
                close=float(last[4]),
                volume=float(last[5]),
                is_closed=False,
            )

            logger.info(
                "Kline bootstrap %s | %d closed bars + 1 in-progress (maxlen=%d)",
                interval, len(buf), KLINE_BUFFER_SIZES[interval],
            )

        self._klines_ready.set()
        logger.info("Kline bootstrap complete")

    # ------------------------------------------------------------------
    # WebSocket stream (level2 + matches in one connection)
    # ------------------------------------------------------------------

    async def _ws_stream(self) -> None:
        """
        Single WebSocket connection for order book and trade data.

        Coinbase allows subscribing to multiple channels on one socket,
        which is simpler and more efficient than the 3-socket Binance model.
        """
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": [self.product_id],
            "channels": ["level2", "matches"],
        }

        while self._running:
            try:
                assert self._session is not None
                async with self._session.ws_connect(
                    COINBASE_WS_URL, heartbeat=30,
                ) as ws:
                    logger.info("Coinbase WS connected: %s", COINBASE_WS_URL)
                    await ws.send_json(subscribe_msg)

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            msg_type = data.get("type")

                            if msg_type == "snapshot":
                                self._handle_book_snapshot(data)
                            elif msg_type == "l2update":
                                self._handle_book_update(data)
                            elif msg_type in ("match", "last_match"):
                                self._handle_trade(data)
                            # "subscriptions", "error", "heartbeat" ignored

                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            logger.warning(
                                "Coinbase WS closed/error — reconnecting"
                            )
                            break

            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Coinbase WS error — reconnecting in 5 s")
                await asyncio.sleep(5)

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    def _handle_book_snapshot(self, data: dict[str, Any]) -> None:
        """Process the initial full order book snapshot from level2 channel."""
        self.book.apply_snapshot(data["bids"], data["asks"])
        self._book_ready.set()

    def _handle_book_update(self, data: dict[str, Any]) -> None:
        """Process incremental book updates from level2 channel."""
        if not self._book_ready.is_set():
            return  # skip updates before initial snapshot
        self.book.apply_update(data["changes"])

    def _handle_trade(self, data: dict[str, Any]) -> None:
        """
        Process a trade (match) from the matches channel.

        Coinbase ``side`` = **maker** order side:
            side="sell" → maker was selling, taker was buying → BUY aggression  → +qty
            side="buy"  → maker was buying, taker was selling → SELL aggression → -qty
        """
        qty = float(data["size"])
        price = float(data["price"])

        signed_qty = qty if data["side"] == "sell" else -qty
        self._trades.append((time.monotonic(), signed_qty))
        self._prune_trades()

        # Build in-progress 1m kline from this trade
        self._update_current_kline(price, qty, data.get("time", ""))

    # ------------------------------------------------------------------
    # Kline construction from trade stream
    # ------------------------------------------------------------------

    def _update_current_kline(
        self, price: float, qty: float, time_str: str,
    ) -> None:
        """Build 1m kline bars incrementally from individual trades."""
        # Parse trade timestamp to determine minute boundary
        try:
            dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            ts_s = int(dt.timestamp())
        except (ValueError, AttributeError):
            ts_s = int(time.time())

        minute_ts_ms = (ts_s // 60) * 60 * 1000

        current = self._current_klines.get("1m")
        if current is not None and current.open_time == minute_ts_ms:
            # Same minute — update in-progress candle
            current.high = max(current.high, price)
            current.low = min(current.low, price)
            current.close = price
            current.volume += qty
        else:
            # New minute — close previous candle, start fresh
            if current is not None:
                current.is_closed = True
                self._kline_buffers["1m"].append(current)
            self._current_klines["1m"] = KlineBar(
                open_time=minute_ts_ms,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=qty,
                is_closed=False,
            )

    def _prune_trades(self) -> None:
        cutoff = time.monotonic() - self.tfi_window
        while self._trades and self._trades[0][0] < cutoff:
            self._trades.popleft()

    # ------------------------------------------------------------------
    # Kline data access
    # ------------------------------------------------------------------

    def get_klines(self, interval: str) -> list[list]:
        """
        Return kline data in standard format (list of lists).
        Index mapping: [0]=open_time_ms, [1]=open, [2]=high, [3]=low,
        [4]=close, [5]=volume.  Includes closed bars + current in-progress.
        """
        buf = self._kline_buffers.get(interval)
        if buf is None:
            return []
        result: list[list] = []
        for bar in buf:
            result.append([
                bar.open_time, str(bar.open), str(bar.high),
                str(bar.low), str(bar.close), str(bar.volume),
            ])
        current = self._current_klines.get(interval)
        if current is not None and not current.is_closed:
            result.append([
                current.open_time, str(current.open), str(current.high),
                str(current.low), str(current.close), str(current.volume),
            ])
        return result

    def get_current_price(self) -> float:
        """Current price from the in-progress 1m kline, or mid_price fallback."""
        current = self._current_klines.get("1m")
        if current is not None:
            return current.close
        return self.book.mid_price

    # ------------------------------------------------------------------
    # Microstructure Metrics
    # ------------------------------------------------------------------

    def compute_obi(self, depth_levels: int = 5) -> float:
        """
        Order Book Imbalance over the top ``depth_levels`` on each side.

            OBI = (bid_qty - ask_qty) / (bid_qty + ask_qty)

        Returns a value in [-1, 1].  Positive = bid-heavy (bullish pressure).
        """
        sorted_bids = sorted(self.book.bids.items(), key=lambda x: -x[0])[:depth_levels]
        sorted_asks = sorted(self.book.asks.items(), key=lambda x: x[0])[:depth_levels]

        bid_qty = sum(q for _, q in sorted_bids)
        ask_qty = sum(q for _, q in sorted_asks)
        total = bid_qty + ask_qty

        if total == 0:
            return 0.0
        return (bid_qty - ask_qty) / total

    def compute_tfi(self) -> float:
        """
        Trade Flow Imbalance over the rolling window.

            TFI = net_signed_volume / total_abs_volume

        Returns a value in [-1, 1].  Positive = net buy aggression.
        """
        self._prune_trades()
        if not self._trades:
            return 0.0

        net = sum(sq for _, sq in self._trades)
        total = sum(abs(sq) for _, sq in self._trades)
        if total == 0:
            return 0.0
        return net / total

    # ------------------------------------------------------------------
    # Oracle Filter
    # ------------------------------------------------------------------

    @staticmethod
    def apply_oracle_filter(
        base_probability: float,
        obi: float,
        tfi: float,
        *,
        obi_weight: float = 0.03,
        tfi_weight: float = 0.05,
        max_shift: float = 0.08,
    ) -> float:
        """
        Shift ``base_probability`` using microstructure signals.

        The shift is a weighted linear combination of OBI and TFI, clamped
        to ``±max_shift`` to prevent the oracle from dominating the ensemble.

            shift = clamp(obi_weight * obi + tfi_weight * tfi, -max_shift, max_shift)
            adjusted = clamp(base_probability + shift, 0.01, 0.99)
        """
        raw_shift = obi_weight * obi + tfi_weight * tfi
        clamped_shift = max(-max_shift, min(max_shift, raw_shift))
        adjusted = max(0.01, min(0.99, base_probability + clamped_shift))

        logger.debug(
            "Oracle filter | base=%.3f obi=%.3f tfi=%.3f shift=%.4f -> %.3f",
            base_probability, obi, tfi, clamped_shift, adjusted,
        )
        return adjusted

    # ------------------------------------------------------------------
    # Convenience: populate MarketState fields
    # ------------------------------------------------------------------

    def get_microstructure_snapshot(self) -> dict[str, float]:
        self._prune_trades()
        trade_count = len(self._trades)
        tfi = self.compute_tfi()
        logger.debug(
            "Microstructure snapshot | trades_in_window=%d tfi=%.3f obi=%.3f",
            trade_count, tfi, self.compute_obi(),
        )
        return {
            "order_book_imbalance": self.compute_obi(),
            "trade_flow_imbalance": tfi,
            "mid_price": self.book.mid_price,
            "spread_bps": self.book.spread_bps,
        }


# ---------------------------------------------------------------------------
# Backward-compatible alias so existing test imports still work
# ---------------------------------------------------------------------------
BinanceOracle = CoinbaseOracle
