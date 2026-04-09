"""
Main Event Loop — Pure 15-Minute BTC Scalper.

Ties together:
- Binance Oracle (WebSocket microstructure + 1m klines)
- Short-Term Signal Engine (order-book + trade-flow)
- Contract Router (15m contract selection)
- Position Sizer (15m-tuned Kelly)
- Kalshi Execution Engine (order placement)

Execution cycle (every 10 seconds):
1. Fetch 1m MarketState from Binance oracle.
2. Evaluate short-term signal engine → 15m signal.
3. Fetch available Kalshi 15m markets.
4. Route to the best 15m contract (or skip if no edge).
5. Size the position via PositionSizer.
6. Execute on Kalshi.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone

import aiohttp
import numpy as np

from coinbase_oracle import CoinbaseOracle
from kalshi_client import KalshiClient, KalshiConfig
from kalshi_ws import KalshiWebSocket
from models import DeferredWatchlist, ExecutionMode, MarketState, PendingOrder, RiskParams, TradeInstruction
from risk import PositionSizer, RiskConfig, RiskManager
from router import ContractRouter
from short_term_engine import ShortTermSignalEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trading_bot.log", mode="a"),
    ],
)
logger = logging.getLogger("main")

# Enable verbose audit logging for the router, risk, and Kalshi WS modules
logging.getLogger("router").setLevel(logging.DEBUG)
logging.getLogger("risk").setLevel(logging.DEBUG)
logging.getLogger("kalshi_ws").setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOOP_INTERVAL_SECONDS = int(os.getenv("LOOP_INTERVAL", "10"))
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
KLINE_LOOKBACK = 100       # 1m candles

# Circuit breaker: max consecutive failures before halting
MAX_CONSECUTIVE_FAILURES = 5

RISK_CFG = RiskConfig(
    initial_equity_cents=int(os.getenv("INITIAL_EQUITY_CENTS", "9730")),  # $97.30
    max_allocation_pct=float(os.getenv("MAX_ALLOC_PCT", "0.05")),
    kelly_fraction=float(os.getenv("KELLY_FRAC", "0.25")),
    max_daily_drawdown_pct=float(os.getenv("MAX_DD_PCT", "0.10")),
    # 8% minimum edge: must clear ~3-4c round-trip fees + spread slippage.
    # Old 4% was trivially passed by the self-referential stink bid formula.
    min_edge_threshold=float(os.getenv("MIN_EDGE", "0.08")),
)

RISK_PARAMS = RiskParams(
    # --- Position sizing (tuned for ~$97 account, 15m only) ---
    max_risk_pct=float(os.getenv("MAX_RISK_PCT", "0.05")),
    max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", "0.10")),
    kelly_fraction=float(os.getenv("KELLY_FRAC", "0.25")),
    max_risk_pct_15m=float(os.getenv("MAX_RISK_PCT_15M", "0.03")),
    max_open_positions=int(os.getenv("MAX_OPEN_POS", "3")),
    # --- 15m thresholds ---
    min_15m_edge=float(os.getenv("MIN_15M_EDGE", "0.08")),
    max_15m_spread_pts=float(os.getenv("MAX_15M_SPREAD", "0.15")),
    min_15m_oi=int(os.getenv("MIN_15M_OI", "0")),
    max_volatility_for_15m=float(os.getenv("MAX_VOL_15M", "0.02")),
)

KALSHI_CFG = KalshiConfig(
    api_key_id=os.getenv("KALSHI_API_KEY_ID", ""),
    private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH", ""),
    fifteen_min_series=os.getenv("KALSHI_15M_SERIES", "KXBTC15M"),
    min_edge=RISK_CFG.min_edge_threshold,
)


# ---------------------------------------------------------------------------
# Market State Builder — 1m only (15m specialist)
# ---------------------------------------------------------------------------

async def _fetch_klines(
    session: aiohttp.ClientSession,
    product_id: str,
    interval: str,
    limit: int,
) -> list[list]:
    """Fetch klines from Coinbase REST API (fallback)."""
    granularity = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}.get(interval, 60)
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    params = {"granularity": str(granularity)}
    async with session.get(url, params=params,
                           timeout=aiohttp.ClientTimeout(total=15)) as resp:
        resp.raise_for_status()
        raw = await resp.json()
    # Coinbase returns reverse-chronological; convert to chronological
    # and remap [ts_s, low, high, open, close, vol] → [ts_ms, open, high, low, close, vol]
    raw = list(reversed(raw))
    result = []
    for k in raw[-limit:]:
        result.append([
            int(k[0]) * 1000, str(k[3]), str(k[2]),
            str(k[1]), str(k[4]), str(k[5]),
        ])
    return result


async def fetch_market_state(
    session: aiohttp.ClientSession,
    oracle: CoinbaseOracle,
    symbol: str = SYMBOL,
) -> MarketState:
    """
    Build a ``MarketState`` from 1m klines + oracle microstructure.
    No higher-TF data needed for pure 15m scalping.
    """
    raw_1m = oracle.get_klines("1m")

    # Fall back to REST if 1m buffer is empty (e.g. during startup)
    if not raw_1m:
        logger.info("Oracle kline buffer empty for 1m — falling back to REST")
        try:
            raw_1m = await _fetch_klines(session, "BTC-USD", "1m", KLINE_LOOKBACK)
        except Exception:
            raw_1m = []
        if not raw_1m:
            raise RuntimeError("No 1m kline data available from WS or REST")

    prices = [float(k[4]) for k in raw_1m]
    highs = [float(k[2]) for k in raw_1m]
    lows = [float(k[3]) for k in raw_1m]
    volumes = [float(k[5]) for k in raw_1m]

    # 15m return: compare current price to price 15 bars ago on 1m
    return_15m = 0.0
    if len(prices) >= 16 and prices[-16] != 0:
        return_15m = (prices[-1] - prices[-16]) / prices[-16]

    # Realised vol from 1m bars (60-bar window ≈ 1 hour)
    volatility_1h = 0.0
    if len(prices) >= 61:
        arr = np.asarray(prices[-61:], dtype=np.float64)
        log_ret = np.diff(np.log(arr))
        volatility_1h = float(np.std(log_ret))

    # Microstructure from oracle
    micro = oracle.get_microstructure_snapshot()

    return MarketState(
        symbol=symbol,
        prices=prices,
        volumes=volumes,
        highs=highs,
        lows=lows,
        order_book_imbalance=micro["order_book_imbalance"],
        trade_flow_imbalance=micro["trade_flow_imbalance"],
        mid_price=micro["mid_price"],
        spread_bps=micro["spread_bps"],
        return_15m=return_15m,
        volatility_1h=volatility_1h,
    )


# ---------------------------------------------------------------------------
# Kalshi Implied Odds Enrichment (15m only)
# ---------------------------------------------------------------------------

def _kalshi_cents(market: dict, field: str) -> int:
    """Convert a Kalshi dollar-string field (e.g. 'yes_ask_dollars') to cents."""
    val = market.get(field) or "0"
    return int(round(float(val) * 100))


def enrich_kalshi_implied(
    state: MarketState,
    fifteen_min_markets: list[dict],
) -> MarketState:
    """Update MarketState with Kalshi implied odds from the best 15m market."""
    if fifteen_min_markets:
        best = min(
            fifteen_min_markets,
            key=lambda m: _kalshi_cents(m, "yes_ask_dollars") - _kalshi_cents(m, "yes_bid_dollars"),
        )
        yes_ask = _kalshi_cents(best, "yes_ask_dollars")
        if yes_ask > 0:
            state.kalshi_15m_implied = yes_ask / 100.0

    return state


# ---------------------------------------------------------------------------
# Trading Cycle
# ---------------------------------------------------------------------------

async def run_trading_cycle(
    session: aiohttp.ClientSession,
    oracle: CoinbaseOracle,
    kalshi: KalshiClient,
    risk: RiskManager,
    sizer: PositionSizer,
    router: ContractRouter,
    signal_engine: ShortTermSignalEngine,
    watchlist: DeferredWatchlist | None = None,
) -> None:
    """
    Single pass of the 15m pipeline:
    signal → route → size → execute.
    """
    cycle_start = datetime.now(timezone.utc)
    logger.info("=== Cycle start %s ===", cycle_start.isoformat())

    # 0a. Reconcile position count with Kalshi (fixes phantom counter)
    actual_positions = await kalshi.fetch_open_position_count()
    if actual_positions >= 0:
        risk.sync_position_count(actual_positions)

    # 0b. Manage pending orders (check fills, cancel stale/expired)
    filled, cancelled, resting = await kalshi.manage_pending_orders()
    if filled or cancelled or resting:
        logger.info(
            "Pending orders | filled=%d cancelled=%d resting=%d",
            filled, cancelled, resting,
        )

    # 0c. Kill switch: cancel all pending if drawdown approaching circuit breaker
    dd_pct = risk._daily_drawdown_pct()
    if dd_pct >= 0.08 and kalshi.resting_count > 0:
        await kalshi.cancel_all_pending(f"drawdown at {dd_pct:.1%}, approaching breaker")

    # 1. Fetch 1m market state (no higher TFs needed)
    try:
        state = await fetch_market_state(session, oracle)
    except Exception:
        logger.error("Market state fetch failed — aborting cycle")
        return

    logger.info(
        "MarketState | mid=%.2f spread=%.1f bps | "
        "1m=%d bars | obi=%.3f tfi=%.3f vol_1h=%.5f ret_15m=%.4f",
        state.mid_price, state.spread_bps,
        len(state.prices),
        state.order_book_imbalance, state.trade_flow_imbalance,
        state.volatility_1h, state.return_15m,
    )

    # 2. Fetch available Kalshi 15m markets only
    fifteen_min_markets = await kalshi.get_active_15m_markets()

    logger.info("Kalshi 15m markets | %d", len(fifteen_min_markets))

    # Enrich state with Kalshi implied odds
    state = enrich_kalshi_implied(state, fifteen_min_markets)

    # 3. Evaluate 15m signal engine only
    short_term_signal = signal_engine.generate(state, oracle, RISK_PARAMS)

    # 4. Route to best 15m contract + stink bid candidates
    #    signal_engine + oracle enable per-contract strike-conditional
    #    probability (GBM model); falls back to generic signal if strike
    #    parsing fails for a given contract.
    best_immediate, stink_candidates, new_watchlist_entries = router.route(
        state=state,
        signal=short_term_signal,
        fifteen_min_markets=fifteen_min_markets,
        signal_engine=signal_engine,
        oracle=oracle,
    )

    # 4a-pre. Record liquidity metrics for execution guard kill switches
    if fifteen_min_markets:
        dead_empty = sum(
            1 for m in fifteen_min_markets
            if not m.get("_has_ws_depth", False)
        )
        dead_empty_pct = dead_empty / len(fifteen_min_markets)
        spreads = []
        for m in fifteen_min_markets:
            ya = float(m.get("yes_ask_dollars") or "0") * 100
            yb = float(m.get("yes_bid_dollars") or "0") * 100
            if ya > 0 and yb > 0:
                spreads.append(ya - yb)
        median_spread = sorted(spreads)[len(spreads) // 2] if spreads else 99.0
        kalshi.guard.record_cycle_liquidity(dead_empty_pct, median_spread)

    # 4a. Feed DEAD skips into watchlist for retry in future cycles
    if watchlist is not None:
        for entry in new_watchlist_entries:
            watchlist.add(entry)
        if new_watchlist_entries:
            logger.info(
                "Watchlist | added %d DEAD entries (total=%d)",
                len(new_watchlist_entries), watchlist.size,
            )

    traded = False

    # 4b. Reevaluate existing resting orders with current signal context
    if kalshi.resting_count > 0:
        reeval_escalated, reeval_cancelled, reeval_resting = (
            await kalshi.reevaluate_resting_orders(
                signal_engine=signal_engine,
                state=state,
                oracle=oracle,
                router=router,
            )
        )
        if reeval_escalated or reeval_cancelled:
            logger.info(
                "Reevaluation | escalated=%d cancelled=%d still_resting=%d",
                reeval_escalated, reeval_cancelled, reeval_resting,
            )

    # 4c. Retry watchlist candidates whose books may have recovered
    if watchlist is not None and best_immediate is None:
        # Build ticker→market lookup from fresh data for retry evaluation
        _market_by_ticker = {m["ticker"]: m for m in fifteen_min_markets}
        for entry in watchlist.get_retry_candidates():
            watchlist.mark_retried(entry.ticker)
            # Use fresh market data if available, fall back to stored snapshot
            fresh_market = _market_by_ticker.get(entry.ticker, entry.market_snapshot)
            # Re-evaluate through the normal contract evaluation
            retry_result = router._evaluate_contract(
                market=fresh_market,
                signal=short_term_signal,
                min_edge=RISK_PARAMS.min_15m_edge,
                max_spread=RISK_PARAMS.max_15m_spread_pts,
                min_oi=RISK_PARAMS.min_15m_oi,
                contract_index=0,
                skip_reasons=[],
                state=state,
                signal_engine=signal_engine,
                oracle=oracle,
            )
            if retry_result is not None:
                logger.info(
                    "Watchlist RETRY | %s recovered (tier=%s edge=%.4f) after %d cycles",
                    entry.ticker, retry_result.liquidity_tier.value,
                    retry_result.edge, entry.retry_count,
                )
                best_immediate = retry_result
                watchlist.remove(entry.ticker)
                break  # Only one immediate trade per cycle

    # 5a. Execute immediate candidate (DEEP/THIN tier) — resting-order-first
    if best_immediate is not None:
        exec_mode = router.determine_execution_mode(best_immediate, state)
        if exec_mode != ExecutionMode.SKIP:
            num_contracts = sizer.size(best_immediate)
            if num_contracts > 0:
                try:
                    result = await kalshi.execute_chosen_contract(
                        best_immediate, num_contracts, execution_mode=exec_mode,
                    )
                    if result is not None:
                        if isinstance(result, TradeInstruction):
                            logger.info(
                                ">>> AGGRESSIVE FILL: %s %s %s x%d @%dc edge=%.4f "
                                "horizon=%s tier=%s",
                                result.action.value, result.side.value,
                                result.ticker, result.contracts,
                                result.limit_price_cents, result.edge,
                                result.horizon, result.liquidity_tier.value,
                            )
                            router.register_position(
                                result.ticker, result.side, result.contracts,
                            )
                        elif isinstance(result, PendingOrder):
                            logger.info(
                                ">>> PASSIVE ORDER: %s %s x%d @%dc edge=%.4f mode=%s",
                                result.side.value, result.ticker,
                                result.contracts, result.limit_price_cents,
                                result.edge_at_placement, result.execution_mode,
                            )
                            router.register_position(
                                result.ticker, result.side, result.contracts,
                            )
                        traded = True
                except Exception:
                    logger.exception("Execution failed for %s", best_immediate.ticker)
            else:
                logger.info("Position sizing returned 0 contracts — skipping immediate")
        else:
            logger.info("Execution mode SKIP for %s", best_immediate.ticker)

    # 5b. Place stink bids for EMPTY tier candidates
    for stink_contract in stink_candidates:
        num_contracts = sizer.size(stink_contract)
        if num_contracts > 0:
            try:
                pending = await kalshi.execute_stink_bid(
                    stink_contract, num_contracts,
                )
                if pending:
                    router.register_position(pending.ticker, pending.side, pending.contracts)
                    traded = True
            except Exception:
                logger.exception("Stink bid failed for %s", stink_contract.ticker)

    if not traded and best_immediate is None and not stink_candidates:
        logger.info("No actionable contract this cycle")

    # Re-sync equity from Kalshi after potential trade
    balance = await kalshi.fetch_balance()
    if balance > 0:
        risk.sync_equity(balance)

    elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
    logger.info("=== Cycle complete in %.2f s ===", elapsed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    logger.info("Initialising pure 15m BTC scalper...")
    logger.info(
        "Config: interval=%ds symbol=%s 15m_series=%s",
        LOOP_INTERVAL_SECONDS, SYMBOL,
        KALSHI_CFG.fifteen_min_series,
    )

    # Components
    risk = RiskManager(RISK_CFG)
    sizer = PositionSizer(RISK_PARAMS, risk)
    oracle = CoinbaseOracle(product_id="BTC-USD")
    router = ContractRouter(RISK_PARAMS)
    kalshi_ws = KalshiWebSocket(KALSHI_CFG)
    kalshi = KalshiClient(KALSHI_CFG, risk, position_sizer=sizer, ws=kalshi_ws)
    signal_engine = ShortTermSignalEngine()
    watchlist = DeferredWatchlist(max_entries=10)

    # Shared HTTP session for REST calls
    session = aiohttp.ClientSession()

    # Graceful shutdown
    shutdown_event = asyncio.Event()
    consecutive_failures = 0

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        # Start subsystems
        await oracle.start()
        await kalshi.start()
        await kalshi_ws.start()

        # Startup reconciliation: query Kalshi for the true live state
        await kalshi.reconcile_on_startup(router)

        # Allow the oracle WebSocket a moment to populate the book
        logger.info("Waiting 3 s for oracle book to populate...")
        await asyncio.sleep(3)

        # Main loop
        while not shutdown_event.is_set():
            try:
                await run_trading_cycle(
                    session, oracle, kalshi, risk, sizer, router, signal_engine,
                    watchlist=watchlist,
                )
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                logger.exception(
                    "Unhandled error in trading cycle (failure %d/%d)",
                    consecutive_failures, MAX_CONSECUTIVE_FAILURES,
                )
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.critical(
                        "CIRCUIT BREAKER: %d consecutive failures — halting bot",
                        consecutive_failures,
                    )
                    break

            # Wait for the next interval or shutdown
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=LOOP_INTERVAL_SECONDS,
                )
            except asyncio.TimeoutError:
                pass  # normal — interval elapsed

    finally:
        logger.info("Shutting down...")
        await kalshi_ws.stop()
        await oracle.stop()
        await kalshi.stop()
        await session.close()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
