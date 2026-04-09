"""
Contract Router — 15-minute BTC contract selector.

Given the short-term signal engine output and live Kalshi 15m market data,
selects the best contract to trade (or None if no edge is actionable)
based on edge magnitude, spread, liquidity, and volatility regime.

Includes verbose audit logging so every skip reason is visible in the log.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from models import (
    Bias, ChosenContract, ExecutionMode, LiquidityTier,
    MarketState, RiskParams, Side, Signal, WatchlistEntry,
)

# Lazy import to avoid circular dependency — resolved at call time
_parse_strike = None

def _get_parse_strike():
    global _parse_strike
    if _parse_strike is None:
        from short_term_engine import parse_strike
        _parse_strike = parse_strike
    return _parse_strike

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kalshi market data helpers
# ---------------------------------------------------------------------------

def _extract_market_metrics(market: dict[str, Any]) -> dict[str, Any]:
    """
    Extract standardised metrics from a Kalshi market dict.

    Kalshi API v2 returns prices as dollar strings:
        - yes_ask_dollars: "0.4200"  (= 42 cents)
        - yes_bid_dollars: "0.4100"  (= 41 cents)
        - open_interest (int)
    """
    yes_ask_str = market.get("yes_ask_dollars") or "0"
    yes_bid_str = market.get("yes_bid_dollars") or "0"

    yes_ask = int(round(float(yes_ask_str) * 100))
    yes_bid = int(round(float(yes_bid_str) * 100))

    ask_prob = yes_ask / 100.0 if yes_ask > 0 else 0.0
    bid_prob = yes_bid / 100.0 if yes_bid > 0 else 0.0
    spread = ask_prob - bid_prob if (yes_ask > 0 and yes_bid > 0) else 0.0
    oi = market.get("open_interest", 0) or 0

    # Parse close_time for expiry management
    close_time_str = market.get("close_time")
    close_time = None
    if close_time_str:
        try:
            close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass

    return {
        "ticker": market.get("ticker", ""),
        "title": market.get("title", ""),
        "yes_ask_cents": yes_ask,
        "yes_bid_cents": yes_bid,
        "ask_prob": ask_prob,
        "bid_prob": bid_prob,
        "spread": spread,
        "open_interest": oi,
        "has_ws_depth": market.get("_has_ws_depth"),
        "ws_liquidity_tier": market.get("_ws_liquidity_tier"),
        "close_time": close_time,
    }


_MIN_STINK_EDGE_CENTS = 8     # minimum edge buffer (in cents) for stink bids on empty books


# ---------------------------------------------------------------------------
# Contract Router
# ---------------------------------------------------------------------------

class ContractRouter:
    """
    Pure 15m route-to-market decision engine.

    Decision logic
    --------------
    1. Score each available 15m contract by ``edge - spread_penalty``.
    2. Apply hard filters: minimum edge, max spread, min OI.
    3. Volatility regime gate: if 1h vol exceeds threshold, skip cycle.
    4. Directional lockout: prevent buying the opposite side of an existing position.
    5. Select the contract with the highest net score.
    """

    # Maximum contracts the bot may accumulate on a single 15m event
    MAX_CONTRACTS_PER_EVENT = 5
    # Minimum seconds between stink bids on the same event ticker
    MIN_SECONDS_BETWEEN_BIDS = 120  # 2 minutes

    def __init__(self, risk_params: RiskParams) -> None:
        self.params = risk_params
        self._first_contract_logged = False
        # Directional lockout: maps ticker -> (Side, expiry_utc) for active positions.
        self._active_positions: dict[str, tuple[Side, datetime | None]] = {}
        # Cumulative contracts per event (event_ticker -> total contracts filled+resting)
        self._event_contracts: dict[str, int] = {}
        # Last stink bid time per event (event_ticker -> utc datetime)
        self._last_bid_time: dict[str, datetime] = {}

    def register_position(self, ticker: str, side: Side, contracts: int = 1) -> None:
        """Record that we have an active position on this ticker."""
        self._active_positions[ticker] = (side, None)
        # Track cumulative exposure per event (event_ticker = ticker minus last segment)
        event_ticker = "-".join(ticker.split("-")[:-1]) if "-" in ticker else ticker
        self._event_contracts[event_ticker] = self._event_contracts.get(event_ticker, 0) + contracts
        self._last_bid_time[event_ticker] = datetime.now(timezone.utc)
        logger.info("Router | Registered active position: %s %s (%d contracts, event total=%d)",
                     side.value, ticker, contracts, self._event_contracts.get(event_ticker, 0))

    def clear_position(self, ticker: str) -> None:
        """Remove a position from the lockout tracker."""
        if ticker in self._active_positions:
            del self._active_positions[ticker]
            logger.info("Router | Cleared active position: %s", ticker)

    def unregister_position(self, ticker: str, contracts: int = 1) -> None:
        """Unregister a cancelled (never-filled) resting order from all trackers."""
        if ticker in self._active_positions:
            del self._active_positions[ticker]
        event_ticker = "-".join(ticker.split("-")[:-1]) if "-" in ticker else ticker
        if event_ticker in self._event_contracts:
            self._event_contracts[event_ticker] = max(
                0, self._event_contracts[event_ticker] - contracts,
            )
        logger.info(
            "Router | Unregistered cancelled order: %s (%d contracts, event total=%d)",
            ticker, contracts, self._event_contracts.get(event_ticker, 0),
        )

    @staticmethod
    def _dynamic_depth_thresholds(vol_1h: float = 0.0) -> tuple[int, int, int]:
        """
        Compute (min_contracts, min_levels, max_spread) for DEEP classification
        adjusted by time-of-day, day-of-week, and realized volatility.
        """
        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday()  # 0=Mon … 6=Sun

        # Time-of-day multiplier (Kalshi BTC peak: 14-22 UTC)
        if 2 <= hour < 8:
            time_mult = 0.50
        elif 8 <= hour < 14:
            time_mult = 0.75
        elif 22 <= hour or hour < 2:
            time_mult = 0.85
        else:
            time_mult = 1.0

        # Day-of-week multiplier
        if weekday == 6:        # Sunday
            day_mult = 0.60
        elif weekday in (0, 5): # Monday, Saturday
            day_mult = 0.80
        else:
            day_mult = 1.0

        # Volatility multiplier (high vol sweeps books → lower bar)
        if vol_1h > 0.015:
            vol_mult = 0.60
        elif vol_1h > 0.008:
            vol_mult = 0.80
        else:
            vol_mult = 1.0

        combined = max(0.30, time_mult * day_mult * vol_mult)

        min_contracts = max(5, int(50 * combined))
        min_levels = max(2, int(4 * combined))
        max_spread = max(3, int(5 / combined))

        return min_contracts, min_levels, max_spread

    def _purge_expired_positions(self) -> None:
        """Remove lockout entries for contracts that have already expired."""
        now = datetime.now(timezone.utc)
        expired = [
            t for t, (_, exp) in self._active_positions.items()
            if exp is not None and exp < now
        ]
        for t in expired:
            del self._active_positions[t]
        if expired:
            logger.info("Router | Purged %d expired position lockouts: %s",
                         len(expired), expired)

    def determine_execution_mode(
        self,
        contract: ChosenContract,
        state: MarketState,
    ) -> ExecutionMode:
        """
        Decide execution strategy based on time remaining, edge quality,
        spread width, liquidity tier, and microstructure adverse-selection risk.

        Default: PASSIVE (post inside spread).
        Escalate when time pressure, adverse flow, or trivial spread
        makes passive posting risky or pointless.

        Tier-specific overrides:
        - THIN_MARGINAL: never AGGRESSIVE (high adverse selection risk)
        - THIN_TRADEABLE: default to MODERATE (midpoint sniper)
        """
        secs_remaining = 900.0
        if contract.close_time:
            secs_remaining = max(
                0, (contract.close_time - datetime.now(timezone.utc)).total_seconds(),
            )

        edge = contract.edge
        min_edge = self.params.min_15m_edge
        spread = contract.spread
        tier = contract.liquidity_tier

        # -------------------------------------------------------------------
        # THIN_MARGINAL: conservative — never aggressive, cap at MODERATE
        # -------------------------------------------------------------------
        if tier == LiquidityTier.THIN_MARGINAL:
            if secs_remaining < 60:
                logger.info(
                    "ExecMode | SKIP: %s THIN_MARGINAL %.0fs left (no aggression allowed)",
                    contract.ticker, secs_remaining,
                )
                return ExecutionMode.SKIP
            if secs_remaining < 180 and edge >= min_edge * 1.5:
                return ExecutionMode.MODERATE
            logger.info(
                "ExecMode | PASSIVE: %s THIN_MARGINAL %.0fs left, edge=%.4f",
                contract.ticker, secs_remaining, edge,
            )
            return ExecutionMode.PASSIVE

        # --- Near expiry (<60 s): aggressive if strong edge, else skip ---
        if secs_remaining < 60:
            if edge >= min_edge * 2.0:
                logger.info(
                    "ExecMode | AGGRESSIVE: %s %.0fs left, edge=%.4f (>2x min)",
                    contract.ticker, secs_remaining, edge,
                )
                return ExecutionMode.AGGRESSIVE
            logger.info(
                "ExecMode | SKIP: %s %.0fs left, edge=%.4f (not enough for late take)",
                contract.ticker, secs_remaining, edge,
            )
            return ExecutionMode.SKIP

        # --- Moderate urgency (1-3 min) ---
        if secs_remaining < 180:
            if edge >= min_edge * 1.5:
                logger.info(
                    "ExecMode | MODERATE: %s %.0fs left, edge=%.4f",
                    contract.ticker, secs_remaining, edge,
                )
                return ExecutionMode.MODERATE
            return ExecutionMode.PASSIVE

        # --- Adverse selection: strong opposing microstructure flow ---
        obi = state.order_book_imbalance
        if contract.side == Side.YES and obi < -0.4:
            if edge < min_edge * 1.5:
                logger.info(
                    "ExecMode | SKIP: %s adverse OBI=%.2f for YES buy, edge=%.4f",
                    contract.ticker, obi, edge,
                )
                return ExecutionMode.SKIP
            logger.info(
                "ExecMode | MODERATE: %s adverse OBI=%.2f but strong edge=%.4f",
                contract.ticker, obi, edge,
            )
            return ExecutionMode.MODERATE
        elif contract.side == Side.NO and obi > 0.4:
            if edge < min_edge * 1.5:
                logger.info(
                    "ExecMode | SKIP: %s adverse OBI=%.2f for NO buy, edge=%.4f",
                    contract.ticker, obi, edge,
                )
                return ExecutionMode.SKIP
            return ExecutionMode.MODERATE

        # --- Narrow spread (<3 c): little to save, use moderate ---
        if 0 < spread < 0.03:
            return ExecutionMode.MODERATE

        # -------------------------------------------------------------------
        # THIN_TRADEABLE: default to MODERATE (midpoint sniper strategy)
        # -------------------------------------------------------------------
        if tier == LiquidityTier.THIN_TRADEABLE:
            logger.info(
                "ExecMode | MODERATE: %s THIN_TRADEABLE, edge=%.4f spread=%.4f (midpoint sniper)",
                contract.ticker, edge, spread,
            )
            return ExecutionMode.MODERATE

        # --- Default: passive ---
        logger.info(
            "ExecMode | PASSIVE: %s %.0fs left, edge=%.4f spread=%.4f",
            contract.ticker, secs_remaining, edge, spread,
        )
        return ExecutionMode.PASSIVE

    def route(
        self,
        state: MarketState,
        signal: Signal,
        fifteen_min_markets: list[dict[str, Any]],
        signal_engine: Any | None = None,
        oracle: Any | None = None,
    ) -> tuple[ChosenContract | None, list[ChosenContract], list[WatchlistEntry]]:
        """
        Select the best 15m contract to trade from available markets.

        Returns (best_immediate, stink_candidates, watchlist_entries):
        - best_immediate: best DEEP/THIN_TRADEABLE/THIN_MARGINAL contract (or None)
        - stink_candidates: up to 5 EMPTY tier contracts sorted by edge
        - watchlist_entries: DEAD contracts with edge, for deferred retry
        """
        candidates: list[ChosenContract] = []
        skip_reasons: list[str] = []
        watchlist_entries: list[WatchlistEntry] = []

        # Purge expired lockout entries before evaluating
        self._purge_expired_positions()

        # Reset per-cycle flag so we log raw data for the first contract
        self._first_contract_logged = False

        # ------------------------------------------------------------------
        # Audit header
        # ------------------------------------------------------------------
        logger.debug(
            "[AUDIT] === Router evaluation start === | "
            "15m_signal: bias=%s p_prob=%.4f edge=%.4f conf=%.3f",
            signal.bias.value, signal.p_prob,
            signal.edge, signal.confidence,
        )
        logger.debug(
            "[AUDIT] Thresholds | 15m: min_edge=%.4f max_spread=%.4f "
            "min_oi=%d max_vol=%.5f",
            self.params.min_15m_edge, self.params.max_15m_spread_pts,
            self.params.min_15m_oi, self.params.max_volatility_for_15m,
        )
        logger.debug(
            "[AUDIT] Markets available | 15m=%d | vol_1h=%.5f",
            len(fifteen_min_markets), state.volatility_1h,
        )

        # ------------------------------------------------------------------
        # Volatility regime gate
        # ------------------------------------------------------------------
        vol_ok = state.volatility_1h <= self.params.max_volatility_for_15m

        if not vol_ok:
            reason = (
                f"15m DISQUALIFIED: vol_1h ({state.volatility_1h:.5f}) "
                f"> max_volatility_for_15m ({self.params.max_volatility_for_15m:.5f})"
            )
            logger.info("Router | %s", reason)
            return None, [], []
        elif not fifteen_min_markets:
            logger.debug("[AUDIT] No 15m markets returned from Kalshi")
        else:
            for i, market in enumerate(fifteen_min_markets):
                candidate = self._evaluate_contract(
                    market=market,
                    signal=signal,
                    min_edge=self.params.min_15m_edge,
                    max_spread=self.params.max_15m_spread_pts,
                    min_oi=self.params.min_15m_oi,
                    contract_index=i,
                    skip_reasons=skip_reasons,
                    state=state,
                    signal_engine=signal_engine,
                    oracle=oracle,
                    watchlist_entries=watchlist_entries,
                )
                if candidate is not None:
                    candidates.append(candidate)

        # ------------------------------------------------------------------
        # Audit summary
        # ------------------------------------------------------------------
        total_evaluated = len(fifteen_min_markets) if vol_ok else 0
        logger.info(
            "Router | Evaluated %d 15m contracts -> %d candidates passed filters",
            total_evaluated, len(candidates),
        )

        if not candidates:
            reason_counts: dict[str, int] = {}
            for r in skip_reasons:
                key = r.split(":")[0] if ":" in r else r
                reason_counts[key] = reason_counts.get(key, 0) + 1
            summary = ", ".join(f"{k}={v}" for k, v in reason_counts.items())
            logger.info(
                "Router | No actionable contracts | skip breakdown: %s",
                summary or "no markets available",
            )
            return None, [], watchlist_entries

        # ------------------------------------------------------------------
        # Split into immediate (DEEP/THIN_*) vs stink bid (EMPTY) candidates
        # ------------------------------------------------------------------
        immediate = [c for c in candidates
                     if c.liquidity_tier in (
                         LiquidityTier.DEEP,
                         LiquidityTier.THIN_TRADEABLE,
                         LiquidityTier.THIN_MARGINAL,
                     )]
        stink = [c for c in candidates
                 if c.liquidity_tier == LiquidityTier.EMPTY]

        best_immediate: ChosenContract | None = None
        if immediate:
            # Liquidity-weighted scoring: prefer deeper books at similar edge
            def _liq_score(c: ChosenContract) -> float:
                tier_w = {
                    LiquidityTier.DEEP: 1.0,
                    LiquidityTier.THIN_TRADEABLE: 0.85,
                    LiquidityTier.THIN_MARGINAL: 0.70,
                }.get(c.liquidity_tier, 0.5)
                return c.edge * (tier_w ** 0.5) - 0.3 * c.spread

            best_immediate = max(immediate, key=_liq_score)
            logger.info(
                "Router | Selected %s tier=%s | side=%s edge=%.4f spread=%.4f "
                "OI=%d p_model=%.4f kalshi_ask=%.4f",
                best_immediate.ticker,
                best_immediate.liquidity_tier.value,
                best_immediate.side.value, best_immediate.edge, best_immediate.spread,
                best_immediate.open_interest, best_immediate.p_model,
                best_immediate.kalshi_ask,
            )

        # Cross-strike routing: if no immediate candidate, check if any EMPTY
        # candidate has a liquid sibling in the same expiry window
        if not immediate and stink:
            for stink_c in stink:
                # Look for an immediate candidate with the same close_time
                for other in candidates:
                    if (
                        other.ticker != stink_c.ticker
                        and other.close_time == stink_c.close_time
                        and other.liquidity_tier in (
                            LiquidityTier.DEEP,
                            LiquidityTier.THIN_TRADEABLE,
                        )
                        and other.edge >= self.params.min_15m_edge
                    ):
                        logger.info(
                            "Router | Cross-strike fallback: %s (EMPTY edge=%.4f) "
                            "→ %s (tier=%s edge=%.4f)",
                            stink_c.ticker, stink_c.edge,
                            other.ticker, other.liquidity_tier.value, other.edge,
                        )
                        best_immediate = other
                        # Remove the replaced stink from the list
                        stink = [s for s in stink if s.ticker != stink_c.ticker]
                        break
                if best_immediate:
                    break

        stink.sort(key=lambda c: c.edge, reverse=True)
        stink_candidates: list[ChosenContract] = stink[:5]

        if stink_candidates:
            logger.info(
                "Router | %d stink bid candidates | best: %s edge=%.4f @%dc",
                len(stink_candidates),
                stink_candidates[0].ticker,
                stink_candidates[0].edge,
                int(stink_candidates[0].kalshi_ask * 100),
            )

        if not best_immediate and not stink_candidates:
            logger.info("Router | Candidates passed filters but none actionable after tier split")
            return None, [], watchlist_entries

        return best_immediate, stink_candidates, watchlist_entries

    # ------------------------------------------------------------------
    # Internal: evaluate a single 15m contract
    # ------------------------------------------------------------------

    def _evaluate_contract(
        self,
        market: dict[str, Any],
        signal: Signal,
        min_edge: float,
        max_spread: float,
        min_oi: int,
        contract_index: int,
        skip_reasons: list[str],
        state: MarketState | None = None,
        signal_engine: Any | None = None,
        oracle: Any | None = None,
        watchlist_entries: list[WatchlistEntry] | None = None,
    ) -> ChosenContract | None:
        """
        Check if a single 15m contract passes all filters.
        Returns a ChosenContract if it does, else None.
        """
        # Log raw API response for the very first contract evaluated
        if not self._first_contract_logged:
            self._first_contract_logged = True
            try:
                raw_json = json.dumps(market, indent=2, default=str)
                logger.debug(
                    "[AUDIT] Raw Kalshi market data (first 15m contract):\n%s",
                    raw_json,
                )
            except Exception:
                logger.debug(
                    "[AUDIT] Raw Kalshi market data (first 15m contract): %s",
                    market,
                )

        metrics = _extract_market_metrics(market)
        ticker = metrics["ticker"]
        ask_prob = metrics["ask_prob"]
        spread = metrics["spread"]
        oi = metrics["open_interest"]

        # Skip markets that are already closed or closing within 30 seconds
        close_time = metrics.get("close_time")
        if close_time is not None:
            secs_to_close = (close_time - datetime.now(timezone.utc)).total_seconds()
            if secs_to_close < 30:
                reason = f"market closing in {secs_to_close:.0f}s (< 30s buffer)"
                skip_reasons.append(reason)
                logger.debug("[AUDIT] SKIP %s: %s", ticker or f"#{contract_index}", reason)
                return None

        if not ticker:
            reason = "empty ticker"
            skip_reasons.append(reason)
            logger.debug("[AUDIT] SKIP contract #%d: %s", contract_index, reason)
            return None

        # ------------------------------------------------------------------
        # Liquidity tier classification (dynamic thresholds)
        # ------------------------------------------------------------------
        has_ws_depth = metrics.get("has_ws_depth")
        ws_tier_str = metrics.get("ws_liquidity_tier")
        vol_1h = state.volatility_1h if state else 0.0
        dyn_contracts, dyn_levels, dyn_spread = self._dynamic_depth_thresholds(vol_1h)

        if has_ws_depth is True:
            # Re-classify with dynamic thresholds using the raw WS tier string
            if ws_tier_str == "deep":
                tier = LiquidityTier.DEEP
            elif ws_tier_str == "thin_tradeable":
                tier = LiquidityTier.THIN_TRADEABLE
            elif ws_tier_str == "thin_marginal":
                tier = LiquidityTier.THIN_MARGINAL
            else:
                # WS returned old-style "thin" — reclassify using dynamic thresholds
                ws_depth = market.get("_ws_total_depth", 0)
                # Promote borderline books: if they pass dynamic DEEP thresholds
                # (which are lowered during off-peak), classify as DEEP
                if ws_depth >= dyn_contracts:
                    tier = LiquidityTier.DEEP
                elif ws_depth >= 10:
                    tier = LiquidityTier.THIN_TRADEABLE
                else:
                    tier = LiquidityTier.THIN_MARGINAL
        elif has_ws_depth is False:
            if metrics["yes_ask_cents"] > 0 or metrics["yes_bid_cents"] > 0:
                tier = LiquidityTier.EMPTY
            else:
                tier = LiquidityTier.DEAD
                reason = f"dead: {ticker} — no WS depth and no listing prices"
                skip_reasons.append(reason)
                logger.debug("[AUDIT] SKIP %s: %s", ticker, reason)
                # Add to watchlist if signal suggests edge potential
                close_time = metrics.get("close_time")
                if (
                    watchlist_entries is not None
                    and close_time is not None
                    and abs(signal.p_prob - 0.5) >= min_edge
                ):
                    side = Side.YES if signal.p_prob >= 0.5 else Side.NO
                    watchlist_entries.append(WatchlistEntry(
                        ticker=ticker,
                        side=side,
                        model_p=signal.p_prob,
                        edge_at_discovery=abs(signal.p_prob - 0.5),
                        close_time=close_time,
                        skip_reason="dead",
                        market_snapshot=market,
                    ))
                return None
        else:
            tier = LiquidityTier.THIN_TRADEABLE  # fallback

        logger.debug(
            "[AUDIT] Tier %s | %s | ws_tier=%s ws_depth=%s dyn_thresholds=(%d,%d,%d)",
            tier.value, ticker, ws_tier_str,
            market.get("_ws_total_depth", "?"),
            dyn_contracts, dyn_levels, dyn_spread,
        )

        # ------------------------------------------------------------------
        # Model probability: per-contract strike-conditional if available,
        # otherwise fall back to generic signal.
        # ------------------------------------------------------------------
        strike_eval: dict[str, float] | None = None
        _parse = _get_parse_strike()
        strike = _parse(market) if _parse else None
        close_time = metrics.get("close_time")

        current_price = 0.0
        if state is not None:
            current_price = state.mid_price if state.mid_price > 0 else (
                state.prices[-1] if state.prices else 0.0
            )

        if (
            strike is not None
            and close_time is not None
            and signal_engine is not None
            and oracle is not None
            and current_price > 0
        ):
            minutes_remaining = max(
                0.0, (close_time - datetime.now(timezone.utc)).total_seconds() / 60.0
            )
            strike_eval = signal_engine.evaluate_contract(
                current_price, strike, minutes_remaining, state, oracle,
            )
            model_p = strike_eval["p_above"]
            logger.debug(
                "[AUDIT] Strike-conditional | %s S₀=%.2f K=%.2f τ=%.1fm "
                "σ=%.6f d₂=%.4f → P(above)=%.4f",
                ticker, current_price, strike, minutes_remaining,
                strike_eval["sigma"], strike_eval["d2"], model_p,
            )
        else:
            model_p = signal.p_prob
            if strike is None and signal_engine is not None:
                logger.debug(
                    "[AUDIT] No strike parsed for %s — using generic p=%.4f",
                    ticker, model_p,
                )

        # ------------------------------------------------------------------
        # Directional lockout: block opposite-side trades on same ticker
        # ------------------------------------------------------------------
        existing_entry = self._active_positions.get(ticker)
        existing_side = existing_entry[0] if existing_entry is not None else None

        # ------------------------------------------------------------------
        # Determine side and compute edge
        # ------------------------------------------------------------------
        if tier == LiquidityTier.EMPTY:
            # Stink bid: derive limit from ACTUAL market ask, NOT model probability.
            # The REST listing prices (yes_ask/no_ask) represent real resting depth
            # even when the WS book hasn't populated yet.
            if model_p >= 0.5:
                side = Side.YES
                # Use the real market ask if available; otherwise skip
                if metrics["yes_ask_cents"] > 0:
                    kalshi_price = metrics["yes_ask_cents"] / 100.0
                else:
                    reason = f"no_ask_empty: {ticker} — no YES ask for stink bid pricing"
                    skip_reasons.append(reason)
                    logger.debug("[AUDIT] SKIP %s: %s", ticker, reason)
                    return None
                edge = model_p - kalshi_price
            else:
                side = Side.NO
                no_model_p = 1.0 - model_p
                # NO ask = yes_bid complement. Use REST no_ask if available.
                no_ask_str = market.get("no_ask_dollars") or "0"
                no_ask_cents = int(round(float(no_ask_str) * 100))
                if no_ask_cents > 0:
                    kalshi_price = no_ask_cents / 100.0
                elif metrics["yes_bid_cents"] > 0:
                    kalshi_price = (100 - metrics["yes_bid_cents"]) / 100.0
                else:
                    reason = f"no_ask_empty: {ticker} — no NO ask for stink bid pricing"
                    skip_reasons.append(reason)
                    logger.debug("[AUDIT] SKIP %s: %s", ticker, reason)
                    return None
                edge = no_model_p - kalshi_price
        else:
            # DEEP/THIN_TRADEABLE/THIN_MARGINAL: use live market prices
            if model_p >= 0.5:
                side = Side.YES
                if metrics["yes_ask_cents"] <= 0:
                    reason = f"no_ask: {ticker} — no YES ask available (yes_ask_cents=0)"
                    skip_reasons.append(reason)
                    logger.debug(
                        "[AUDIT] SKIP %s: No YES ask available "
                        "(yes_ask_cents=%d, yes_bid_cents=%d) — cannot price YES side",
                        ticker,
                        metrics["yes_ask_cents"], metrics["yes_bid_cents"],
                    )
                    return None
                edge = model_p - ask_prob
                kalshi_price = ask_prob
            else:
                side = Side.NO
                if metrics["yes_bid_cents"] > 0:
                    no_implied = (100 - metrics["yes_bid_cents"]) / 100.0
                else:
                    reason = (
                        f"no_liquidity: {ticker} — no YES bid available "
                        f"(yes_bid={metrics['yes_bid_cents']}, yes_ask={metrics['yes_ask_cents']}) "
                        f"— cannot price NO side"
                    )
                    skip_reasons.append(reason)
                    logger.debug("[AUDIT] SKIP %s: %s", ticker, reason)
                    return None
                edge = (1.0 - model_p) - no_implied
                kalshi_price = no_implied

        # ------------------------------------------------------------------
        # Directional lockout enforcement
        # ------------------------------------------------------------------
        if existing_side is not None and existing_side != side:
            reason = (
                f"lockout: {ticker} — already holding {existing_side.value}, "
                f"refusing opposite side {side.value} (anti self-hedge)"
            )
            skip_reasons.append(reason)
            logger.debug("[AUDIT] SKIP %s: %s", ticker, reason)
            return None

        # ------------------------------------------------------------------
        # Per-event cumulative exposure cap
        # ------------------------------------------------------------------
        event_ticker = market.get("event_ticker") or ("-".join(ticker.split("-")[:-1]) if "-" in ticker else ticker)
        event_total = self._event_contracts.get(event_ticker, 0)
        if event_total >= self.MAX_CONTRACTS_PER_EVENT:
            reason = (
                f"event_cap: {ticker} — event {event_ticker} already has "
                f"{event_total} contracts (max {self.MAX_CONTRACTS_PER_EVENT})"
            )
            skip_reasons.append(reason)
            logger.debug("[AUDIT] SKIP %s: %s", ticker, reason)
            return None

        # Rate-limit stink bids on the same event
        if tier == LiquidityTier.EMPTY:
            last_bid = self._last_bid_time.get(event_ticker)
            if last_bid is not None:
                elapsed = (datetime.now(timezone.utc) - last_bid).total_seconds()
                if elapsed < self.MIN_SECONDS_BETWEEN_BIDS:
                    reason = (
                        f"rate_limit: {ticker} — last bid on {event_ticker} was "
                        f"{elapsed:.0f}s ago (min {self.MIN_SECONDS_BETWEEN_BIDS}s)"
                    )
                    skip_reasons.append(reason)
                    logger.debug("[AUDIT] SKIP %s: %s", ticker, reason)
                    return None

        # ------------------------------------------------------------------
        # Hard filters
        # ------------------------------------------------------------------

        # 1. Minimum edge
        if edge < min_edge:
            reason = (
                f"low_edge: {ticker} — edge ({edge:.4f}) < "
                f"min_15m_edge ({min_edge:.4f}) | "
                f"side={side.value} model_p={model_p:.4f} "
                f"kalshi_price={kalshi_price:.4f}"
            )
            skip_reasons.append(reason)
            logger.debug(
                "[AUDIT] SKIP %s: Edge %.4f < min %.4f | "
                "side=%s model_p=%.4f kalshi=%.4f yes_ask=%dc yes_bid=%dc",
                ticker, edge, min_edge,
                side.value, model_p, kalshi_price,
                metrics["yes_ask_cents"], metrics["yes_bid_cents"],
            )
            return None

        # 2. Maximum spread (skip for EMPTY books — spread is undefined)
        if tier != LiquidityTier.EMPTY and spread > max_spread:
            reason = (
                f"wide_spread: {ticker} — spread ({spread:.4f}) > "
                f"max_15m_spread ({max_spread:.4f})"
            )
            skip_reasons.append(reason)
            logger.debug(
                "[AUDIT] SKIP %s: Spread %.4f > max %.4f | "
                "yes_ask=%dc yes_bid=%dc",
                ticker, spread, max_spread,
                metrics["yes_ask_cents"], metrics["yes_bid_cents"],
            )
            return None

        # 3. Minimum open interest (skip for EMPTY books — OI is always 0)
        if tier != LiquidityTier.EMPTY and oi < min_oi:
            reason = (
                f"low_oi: {ticker} — OI ({oi}) < min_15m_oi ({min_oi})"
            )
            skip_reasons.append(reason)
            logger.debug("[AUDIT] SKIP %s: OI %d < min %d", ticker, oi, min_oi)
            return None

        # ------------------------------------------------------------------
        # Contract passed all filters — must be 15m
        # ------------------------------------------------------------------
        logger.debug(
            "[AUDIT] PASS %s tier=%s: side=%s edge=%.4f spread=%.4f OI=%d "
            "p_model=%.4f kalshi=%.4f",
            ticker, tier.value, side.value, edge, spread, oi,
            model_p if side == Side.YES else 1.0 - model_p,
            kalshi_price,
        )

        return ChosenContract(
            ticker=ticker,
            horizon="15m",
            side=side,
            p_model=model_p if side == Side.YES else 1.0 - model_p,
            kalshi_ask=kalshi_price,
            edge=edge,
            spread=spread if tier != LiquidityTier.EMPTY else 0.0,
            open_interest=oi,
            liquidity_tier=tier,
            close_time=metrics.get("close_time"),
            metadata={
                "title": metrics["title"],
                "signal_bias": signal.bias.value,
                "signal_confidence": signal.confidence,
                "yes_ask_cents": metrics["yes_ask_cents"],
                "yes_bid_cents": metrics["yes_bid_cents"],
                "liquidity_tier": tier.value,
                "strike": strike,
                "strike_eval": strike_eval,
            },
        )
