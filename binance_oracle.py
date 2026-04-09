"""
Binance Microstructure Oracle.

Maintains a real-time L2 order book via WebSocket depth stream, computes
Order Book Imbalance (OBI) and Trade Flow Imbalance (TFI), and provides
an ``apply_oracle_filter`` that shifts an ensemble base probability toward
or away from the bullish direction based on microstructure alignment.

Also maintains real-time kline (candlestick) buffers via a combined
WebSocket kline stream, replacing the REST kline fetches in the main loop.

Data flow
---------
1. Fetch REST depth snapshot  →  seed local book
2. Subscribe to ``@depth@100ms`` diff stream  →  apply deltas (``lastUpdateId``)
3. Subscribe to ``@aggTrade`` stream  →  accumulate recent trades for TFI
4. Bootstrap historical klines via REST  →  seed kline buffers
5. Subscribe to combined kline stream  →  update buffers in real time
6. On demand: compute OBI, TFI, apply oracle filter; serve kline data.
"""

from __future__ import annotations
import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BINANCE_WS_BASE = "wss://stream.binance.us:9443/ws"
BINANCE_STREAM_BASE = "wss://stream.binance.us:9443/stream"
BINANCE_REST_BASE = "https://api.binance.us"

DEFAULT_SYMBOL = "btcusdt"
DEPTH_SNAPSHOT_LIMIT = 100          # levels
TFI_WINDOW_SECONDS = 300.0         # rolling window for trade flow (5 min — Binance US has low volume)

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
    open_time: int
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
    """Sorted L2 book maintained from REST snapshot + WS diffs."""

    bids: dict[float, float] = field(default_factory=dict)  # price -> qty
    asks: dict[float, float] = field(default_factory=dict)
    last_update_id: int = 0

    def apply_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.last_update_id = snapshot["lastUpdateId"]
        self.bids = {float(p): float(q) for p, q in snapshot["bids"]}
        self.asks = {float(p): float(q) for p, q in snapshot["asks"]}
        logger.info(
            "Order book snapshot applied | lastUpdateId=%d bids=%d asks=%d",
            self.last_update_id, len(self.bids), len(self.asks),
        )

    def apply_diff(self, event: dict[str, Any]) -> bool:
        """
        Apply a depth diff event.  Returns False if the event is stale.

        Binance diff semantics:
        - ``U`` = first update ID in event
        - ``u`` = final update ID in event
        - Drop event if ``u <= lastUpdateId``
        - First valid event must satisfy ``U <= lastUpdateId + 1 <= u``
        """
        first_id: int = event["U"]
        final_id: int = event["u"]

        if final_id <= self.last_update_id:
            return False  # stale

        for price_s, qty_s in event["b"]:
            p, q = float(price_s), float(qty_s)
            if q == 0.0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = q

        for price_s, qty_s in event["a"]:
            p, q = float(price_s), float(qty_s)
            if q == 0.0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = q

        self.last_update_id = final_id
        return True

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
# Binance Oracle
# ---------------------------------------------------------------------------

class BinanceOracle:
    """
    Async WebSocket client that maintains a local order book and computes
    microstructure metrics in real time.
    """

    def __init__(
        self,
        symbol: str = DEFAULT_SYMBOL,
        tfi_window: float = TFI_WINDOW_SECONDS,
    ) -> None:
        self.symbol = symbol.lower()
        self.tfi_window = tfi_window

        self.book = LocalOrderBook()
        # Rolling buffer of (timestamp, signed_qty) for TFI
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Fetch the REST snapshot, bootstrap klines, then launch WS listeners."""
        self._session = aiohttp.ClientSession()
        self._running = True

        await self._fetch_depth_snapshot()
        await self._bootstrap_klines()

        self._tasks = [
            asyncio.create_task(self._depth_stream(), name="depth_stream"),
            asyncio.create_task(self._trade_stream(), name="trade_stream"),
            asyncio.create_task(self._kline_stream(), name="kline_stream"),
        ]
        logger.info("BinanceOracle started for %s", self.symbol.upper())

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        if self._session:
            await self._session.close()
        logger.info("BinanceOracle stopped")

    # ------------------------------------------------------------------
    # REST snapshot
    # ------------------------------------------------------------------

    async def _fetch_depth_snapshot(self) -> None:
        url = f"{BINANCE_REST_BASE}/api/v3/depth"
        params = {"symbol": self.symbol.upper(), "limit": DEPTH_SNAPSHOT_LIMIT}
        assert self._session is not None
        try:
            async with self._session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                data = await resp.json()
                self.book.apply_snapshot(data)
        except Exception:
            logger.exception("Failed to fetch depth snapshot")
            raise

    # ------------------------------------------------------------------
    # WebSocket streams
    # ------------------------------------------------------------------

    async def _depth_stream(self) -> None:
        """Subscribe to ``<symbol>@depth@100ms`` and apply diffs."""
        stream = f"{BINANCE_WS_BASE}/{self.symbol}@depth@100ms"
        while self._running:
            try:
                assert self._session is not None
                async with self._session.ws_connect(stream, heartbeat=30) as ws:
                    logger.info("Depth WS connected: %s", stream)
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            event = msg.json()
                            self.book.apply_diff(event)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            logger.warning("Depth WS closed/error — reconnecting")
                            break
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Depth stream error — reconnecting in 5 s")
                await asyncio.sleep(5)

    async def _trade_stream(self) -> None:
        """Subscribe to ``<symbol>@aggTrade`` and buffer recent trades."""
        stream = f"{BINANCE_WS_BASE}/{self.symbol}@aggTrade"
        while self._running:
            try:
                assert self._session is not None
                async with self._session.ws_connect(stream, heartbeat=30) as ws:
                    logger.info("AggTrade WS connected: %s", stream)
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            trade = msg.json()
                            qty = float(trade["q"])
                            # Binance: ``m`` = True → buyer is market maker → SELL aggressor
                            signed_qty = -qty if trade["m"] else qty
                            self._trades.append((time.monotonic(), signed_qty))
                            self._prune_trades()
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            logger.warning("Trade WS closed/error — reconnecting")
                            break
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Trade stream error — reconnecting in 5 s")
                await asyncio.sleep(5)

    def _prune_trades(self) -> None:
        cutoff = time.monotonic() - self.tfi_window
        while self._trades and self._trades[0][0] < cutoff:
            self._trades.popleft()

    # ------------------------------------------------------------------
    # Kline bootstrap (REST backfill)
    # ------------------------------------------------------------------

    async def _bootstrap_klines(self) -> None:
        """
        Fetch historical klines via REST and populate the deque buffers.
        WebSocket only sends the current kline forward, not history.
        """
        url = f"{BINANCE_REST_BASE}/api/v3/klines"
        symbol_upper = self.symbol.upper()
        assert self._session is not None

        async def _fetch_one(interval: str, limit: int) -> tuple[str, list]:
            params = {"symbol": symbol_upper, "interval": interval, "limit": limit}
            try:
                async with self._session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    resp.raise_for_status()
                    return interval, await resp.json()
            except Exception:
                logger.exception("Failed to bootstrap klines for %s", interval)
                return interval, []

        results = await asyncio.gather(*(
            _fetch_one(iv, sz) for iv, sz in KLINE_BUFFER_SIZES.items()
        ))

        for interval, raw_klines in results:
            buf = self._kline_buffers[interval]
            if not raw_klines:
                continue
            # All bars except the last are definitely closed.
            # The last bar from REST is the in-progress candle — store it in
            # _current_klines instead of the buffer so we don't create a
            # duplicate when the WS later appends the real closed version.
            for k in raw_klines[:-1]:
                buf.append(KlineBar(
                    open_time=int(k[0]),
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    is_closed=True,
                ))
            last = raw_klines[-1]
            self._current_klines[interval] = KlineBar(
                open_time=int(last[0]),
                open=float(last[1]),
                high=float(last[2]),
                low=float(last[3]),
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
    # Kline WebSocket stream
    # ------------------------------------------------------------------

    async def _kline_stream(self) -> None:
        """Subscribe to combined kline streams for all intervals."""
        streams = "/".join(f"{self.symbol}@kline_{iv}" for iv in KLINE_BUFFER_SIZES)
        url = f"{BINANCE_STREAM_BASE}?streams={streams}"

        while self._running:
            try:
                assert self._session is not None
                async with self._session.ws_connect(url, heartbeat=30) as ws:
                    logger.info("Kline WS connected: %s", url)
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            payload = msg.json()
                            data = payload.get("data", payload)
                            k = data.get("k")
                            if k is None:
                                continue
                            interval = k["i"]
                            bar = KlineBar(
                                open_time=int(k["t"]),
                                open=float(k["o"]),
                                high=float(k["h"]),
                                low=float(k["l"]),
                                close=float(k["c"]),
                                volume=float(k["v"]),
                                is_closed=bool(k["x"]),
                            )
                            self._current_klines[interval] = bar
                            if bar.is_closed:
                                buf = self._kline_buffers.get(interval)
                                if buf is not None:
                                    buf.append(bar)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            logger.warning("Kline WS closed/error — reconnecting")
                            break
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Kline stream error — reconnecting in 5 s")
                await asyncio.sleep(5)

    # ------------------------------------------------------------------
    # Kline data access
    # ------------------------------------------------------------------

    def get_klines(self, interval: str) -> list[list]:
        """
        Return kline data in Binance REST format (list of lists).
        Index mapping: [0]=open_time, [1]=open, [2]=high, [3]=low,
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

        Parameters
        ----------
        base_probability : Pre-oracle probability from ensemble.
        obi : Order Book Imbalance in [-1, 1].
        tfi : Trade Flow Imbalance in [-1, 1].
        obi_weight : Sensitivity to order book imbalance.
        tfi_weight : Sensitivity to trade flow imbalance.
        max_shift : Hard cap on the absolute probability shift.
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
