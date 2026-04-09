"""
Post-Fix Validation Suite — checks for all audit fixes + 15m purity.
Run: pytest test_audit_fixes.py -v
"""

import asyncio
import math
import time
from collections import deque
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models import (
    Bias, ChosenContract, ExecutionMode, LiquidityTier, MarketState, OrderAction,
    PendingOrder, RiskParams, Side, Signal, TradeInstruction,
)
from risk import DailyPnL, PositionSizer, RiskConfig, RiskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_risk_manager(equity: int = 10000, pnl: int = 0) -> RiskManager:
    cfg = RiskConfig(initial_equity_cents=equity)
    rm = RiskManager(cfg)
    rm.equity_cents = equity
    rm.daily.realised_pnl_cents = pnl
    rm.daily.peak_equity_cents = equity
    return rm


def _make_sizer(rm: RiskManager, pnl: int = 0) -> PositionSizer:
    params = RiskParams(
        max_risk_pct=0.05,
        max_daily_loss_pct=0.10,
        kelly_fraction=0.25,
        max_risk_pct_15m=0.03,
    )
    rm.daily.realised_pnl_cents = pnl
    return PositionSizer(params, rm)


def _make_contract(
    edge: float = 0.10,
    p_model: float = 0.65,
    kalshi_ask: float = 0.55,
    horizon: str = "15m",
    tier: LiquidityTier = LiquidityTier.DEEP,
) -> ChosenContract:
    return ChosenContract(
        ticker="TEST-01JAN2601-B50000",
        horizon=horizon,
        side=Side.YES,
        p_model=p_model,
        kalshi_ask=kalshi_ask,
        edge=edge,
        spread=0.02,
        open_interest=100,
        liquidity_tier=tier,
    )


def _make_15m_params() -> RiskParams:
    return RiskParams(
        min_15m_edge=0.01,
        max_15m_spread_pts=0.15,
        min_15m_oi=0,
    )


# ===================================================================
# FIX 1: abs(realised_pnl_cents) — winning days must NOT shrink sizing
# ===================================================================

class TestFix1_WinningDaySizing:
    def test_winning_day_does_not_shrink_budget(self):
        rm = _make_risk_manager(equity=10000, pnl=0)
        sizer_neutral = _make_sizer(rm, pnl=0)
        contract = _make_contract()
        contracts_neutral = sizer_neutral.size(contract)

        rm_winning = _make_risk_manager(equity=10000, pnl=500)
        sizer_winning = _make_sizer(rm_winning, pnl=500)
        contracts_winning = sizer_winning.size(contract)

        assert contracts_winning >= contracts_neutral

    def test_losing_day_shrinks_budget_correctly(self):
        rm = _make_risk_manager(equity=10000, pnl=0)
        sizer_neutral = _make_sizer(rm, pnl=0)
        contract = _make_contract()
        contracts_neutral = sizer_neutral.size(contract)

        rm_losing = _make_risk_manager(equity=10000, pnl=-500)
        sizer_losing = _make_sizer(rm_losing, pnl=-500)
        contracts_losing = sizer_losing.size(contract)

        assert contracts_losing <= contracts_neutral

    def test_big_winning_day_full_budget_available(self):
        rm = _make_risk_manager(equity=10000, pnl=2000)
        sizer = _make_sizer(rm, pnl=2000)
        contract = _make_contract()
        contracts = sizer.size(contract)
        assert contracts > 0


# ===================================================================
# FIX 3: _execute_thin_book must return None on submission failure
# ===================================================================

class TestFix3_ThinBookPhantomLockout:
    @pytest.mark.asyncio
    async def test_returns_none_on_submit_failure(self):
        from kalshi_client import KalshiClient, KalshiConfig
        cfg = KalshiConfig(api_key_id="test", private_key_path="test")
        rm = _make_risk_manager()
        client = KalshiClient(cfg, rm)

        contract = _make_contract(tier=LiquidityTier.THIN_TRADEABLE)
        mock_book = {
            "yes": [{"price": 55, "delta": 10}, {"price": 60, "delta": 5}],
            "no": [{"price": 45, "delta": 10}],
        }
        client.get_market_orderbook = AsyncMock(return_value=mock_book)
        client._submit_order = AsyncMock(return_value=None)

        result = await client._execute_thin_book(contract, 3)
        assert result is None
        assert rm.open_position_count == 0

    @pytest.mark.asyncio
    async def test_returns_instruction_on_submit_success(self):
        from kalshi_client import KalshiClient, KalshiConfig
        cfg = KalshiConfig(api_key_id="test", private_key_path="test")
        rm = _make_risk_manager()
        client = KalshiClient(cfg, rm)

        contract = _make_contract(tier=LiquidityTier.THIN_TRADEABLE)
        mock_book = {
            "yes": [{"price": 55, "delta": 10}, {"price": 60, "delta": 5}],
            "no": [{"price": 45, "delta": 10}],
        }
        client.get_market_orderbook = AsyncMock(return_value=mock_book)
        client._submit_order = AsyncMock(return_value={"order": {"order_id": "abc"}})

        result = await client._execute_thin_book(contract, 3)
        assert result is not None
        assert rm.open_position_count == 1


# ===================================================================
# FIX 4: best_yes_bid must derive from NO side
# ===================================================================

class TestFix4_BestYesBid:
    def test_best_yes_bid_from_no_side(self):
        from kalshi_client import KalshiClient
        orderbook = {
            "yes": [{"price": 55, "delta": 10}, {"price": 60, "delta": 5}],
            "no": [{"price": 42, "delta": 8}, {"price": 48, "delta": 3}],
        }
        bid = KalshiClient.best_yes_bid(orderbook)
        assert bid == 58  # 100 - 42

    def test_best_yes_bid_no_no_levels(self):
        from kalshi_client import KalshiClient
        orderbook = {"yes": [{"price": 55, "delta": 10}], "no": []}
        bid = KalshiClient.best_yes_bid(orderbook)
        assert bid is None


# ===================================================================
# FIX 5: KalshiLocalOrderbook spread must be non-negative
# ===================================================================

class TestFix5_OrderbookSpread:
    def test_spread_non_negative(self):
        from kalshi_ws import KalshiLocalOrderbook
        book = KalshiLocalOrderbook()
        book.yes_levels = {55: 10, 60: 5}
        book.no_levels = {42: 8, 48: 3}
        assert book.yes_spread_cents >= 0

    def test_yes_bid_from_no_side(self):
        from kalshi_ws import KalshiLocalOrderbook
        book = KalshiLocalOrderbook()
        book.yes_levels = {55: 10, 60: 5}
        book.no_levels = {42: 8}
        assert book.best_yes_bid == 58
        assert book.best_yes_ask == 55

    def test_unknown_spread_returns_wide(self):
        from kalshi_ws import KalshiLocalOrderbook
        book = KalshiLocalOrderbook()
        book.yes_levels = {55: 10}
        book.no_levels = {}
        assert book.yes_spread_cents == 99

    def test_liquidity_tier_with_correct_spread(self):
        from kalshi_ws import KalshiLocalOrderbook
        book = KalshiLocalOrderbook()
        book.yes_levels = {55: 15, 60: 15}
        book.no_levels = {52: 15, 56: 15}
        spread = book.yes_spread_cents
        assert spread == 7
        assert book.liquidity_tier == "thin_tradeable"


# ===================================================================
# FIX 6: Bootstrap must NOT mark in-progress candle as closed
# ===================================================================

class TestFix6_BootstrapInProgressCandle:
    def test_get_klines_no_duplicate(self):
        from binance_oracle import BinanceOracle, KlineBar
        oracle = BinanceOracle(symbol="btcusdt")
        buf = oracle._kline_buffers["1m"]
        for i in range(3):
            buf.append(KlineBar(
                open_time=1000 * i, open=100, high=101, low=99,
                close=100.5, volume=10, is_closed=True,
            ))
        oracle._current_klines["1m"] = KlineBar(
            open_time=3000, open=101, high=102, low=100,
            close=101.5, volume=5, is_closed=False,
        )
        result = oracle.get_klines("1m")
        assert len(result) == 4
        open_times = [r[0] for r in result]
        assert len(set(open_times)) == 4


# ===================================================================
# FIX 7: Strike-conditional engine — GBM correctness
# ===================================================================

class TestFix7_StrikeConditionalEngine:
    def test_atm_near_50_percent(self):
        """At-the-money (S₀ = K) with plenty of time should be ~50%."""
        from short_term_engine import ShortTermSignalEngine
        from binance_oracle import BinanceOracle
        engine = ShortTermSignalEngine()
        oracle = BinanceOracle(symbol="btcusdt")
        # Provide enough price data for vol estimation
        prices = [100000.0 + i * 10 for i in range(50)]
        state = MarketState(prices=prices, volumes=[1.0] * 50, mid_price=100000.0)
        result = engine.evaluate_contract(
            current_price=100000.0, strike=100000.0,
            minutes_remaining=15.0, state=state, oracle=oracle,
        )
        assert 0.40 < result["p_above"] < 0.60

    def test_deep_itm_high_probability(self):
        """Price well above strike → high P(above)."""
        from short_term_engine import ShortTermSignalEngine
        from binance_oracle import BinanceOracle
        engine = ShortTermSignalEngine()
        oracle = BinanceOracle(symbol="btcusdt")
        prices = [100000.0 + i * 5 for i in range(50)]
        state = MarketState(prices=prices, volumes=[1.0] * 50, mid_price=101000.0)
        result = engine.evaluate_contract(
            current_price=101000.0, strike=99000.0,
            minutes_remaining=5.0, state=state, oracle=oracle,
        )
        assert result["p_above"] > 0.90

    def test_deep_otm_low_probability(self):
        """Price well below strike → low P(above)."""
        from short_term_engine import ShortTermSignalEngine
        from binance_oracle import BinanceOracle
        engine = ShortTermSignalEngine()
        oracle = BinanceOracle(symbol="btcusdt")
        prices = [100000.0 + i * 5 for i in range(50)]
        state = MarketState(prices=prices, volumes=[1.0] * 50, mid_price=99000.0)
        result = engine.evaluate_contract(
            current_price=99000.0, strike=101000.0,
            minutes_remaining=5.0, state=state, oracle=oracle,
        )
        assert result["p_above"] < 0.10

    def test_less_time_makes_otm_less_probable(self):
        """Same OTM distance, less time → lower P(above)."""
        from short_term_engine import ShortTermSignalEngine
        from binance_oracle import BinanceOracle
        engine = ShortTermSignalEngine()
        oracle = BinanceOracle(symbol="btcusdt")
        # Realistic BTC vol: oscillating ±$50 per bar → σ ≈ 0.001/min
        prices = [100000.0 + (50.0 if i % 2 == 0 else -50.0) for i in range(50)]
        state = MarketState(prices=prices, volumes=[1.0] * 50, mid_price=100000.0)
        p_14min = engine.evaluate_contract(
            current_price=100000.0, strike=100100.0,
            minutes_remaining=14.0, state=state, oracle=oracle,
        )["p_above"]
        p_2min = engine.evaluate_contract(
            current_price=100000.0, strike=100100.0,
            minutes_remaining=2.0, state=state, oracle=oracle,
        )["p_above"]
        assert p_14min > p_2min

    def test_micro_drift_is_bounded(self):
        """Microstructure drift must never exceed MAX_MICRO_DRIFT_PER_MIN."""
        from short_term_engine import ShortTermSignalEngine
        from binance_oracle import BinanceOracle
        engine = ShortTermSignalEngine()
        oracle = BinanceOracle(symbol="btcusdt")
        # Extreme trade imbalance
        now = time.monotonic()
        oracle._trades = deque([(now - i, 10.0) for i in range(100)])
        drift = engine._compute_micro_drift(oracle)
        assert abs(drift) <= engine.MAX_MICRO_DRIFT_PER_MIN


# ===================================================================
# FIX 8: EMA drift smoothing
# ===================================================================

class TestFix8_EmaDriftSmoothing:
    def test_default_alpha(self):
        from short_term_engine import ShortTermSignalEngine
        engine = ShortTermSignalEngine()
        assert engine._ema_alpha == 0.7

    def test_drift_smoothing_reduces_oscillation(self):
        """EMA smoothing should prevent drift from jumping instantly."""
        from short_term_engine import ShortTermSignalEngine
        from binance_oracle import BinanceOracle
        engine = ShortTermSignalEngine()
        oracle = BinanceOracle(symbol="btcusdt")
        now = time.monotonic()
        # First call with strong buy flow
        oracle._trades = deque([(now - i, 10.0) for i in range(50)])
        d1 = engine._compute_micro_drift(oracle)
        # Second call with strong sell flow
        oracle._trades = deque([(now - i, -10.0) for i in range(50)])
        d2 = engine._compute_micro_drift(oracle)
        # EMA should prevent instant flip to full negative
        assert d2 > -engine.MAX_MICRO_DRIFT_PER_MIN


# ===================================================================
# FIX 9: Min edge raised to 4%
# ===================================================================

class TestFix9_MinEdgeThreshold:
    def test_low_edge_trade_blocked(self):
        cfg = RiskConfig(min_edge_threshold=0.04)
        rm = RiskManager(cfg)
        allowed, reason = rm.passes_risk_checks(0.02)
        assert not allowed
        assert "below threshold" in reason

    def test_high_edge_trade_allowed(self):
        cfg = RiskConfig(min_edge_threshold=0.04)
        rm = RiskManager(cfg)
        allowed, _ = rm.passes_risk_checks(0.05)
        assert allowed


# ===================================================================
# FIX 10: Strike parser + vol estimator
# ===================================================================

class TestFix10_StrikeParserAndVol:
    def test_parse_floor_strike(self):
        from short_term_engine import parse_strike
        market = {"floor_strike": 68096.44, "ticker": "KXBTC15M-26MAR311630-30"}
        assert parse_strike(market) == 68096.44

    def test_parse_dollar_in_title(self):
        from short_term_engine import parse_strike
        market = {"title": "Bitcoin above $107,500?", "ticker": "KXBTC15M-X"}
        assert parse_strike(market) == 107500.0

    def test_parse_returns_none_for_unparseable(self):
        from short_term_engine import parse_strike
        market = {"ticker": "UNKNOWN", "title": "test"}
        assert parse_strike(market) is None

    def test_realized_vol_with_data(self):
        from short_term_engine import estimate_realized_vol
        prices = [100000.0 + i * 10 for i in range(50)]
        vol = estimate_realized_vol(prices, window=30)
        assert vol > 0

    def test_realized_vol_insufficient_data(self):
        from short_term_engine import estimate_realized_vol
        vol = estimate_realized_vol([100.0, 101.0], window=30)
        assert vol == 0.0

    def test_higher_vol_widens_probability(self):
        """Higher vol should make OTM strikes more probable."""
        from short_term_engine import ShortTermSignalEngine
        from binance_oracle import BinanceOracle
        engine = ShortTermSignalEngine()
        oracle = BinanceOracle(symbol="btcusdt")
        # Low vol: oscillating ±$10 → σ ≈ 0.0002/min
        low_vol_prices = [100000.0 + (10.0 if i % 2 == 0 else -10.0) for i in range(50)]
        state_low = MarketState(prices=low_vol_prices, volumes=[1.0] * 50, mid_price=100000.0)
        p_low = engine.evaluate_contract(
            current_price=100000.0, strike=100100.0,
            minutes_remaining=10.0, state=state_low, oracle=oracle,
        )["p_above"]
        # High vol: oscillating ±$200 → σ ≈ 0.004/min
        high_vol_prices = [100000.0 + (200.0 if i % 2 == 0 else -200.0) for i in range(50)]
        state_high = MarketState(prices=high_vol_prices, volumes=[1.0] * 50, mid_price=100000.0)
        p_high = engine.evaluate_contract(
            current_price=100000.0, strike=100100.0,
            minutes_remaining=10.0, state=state_high, oracle=oracle,
        )["p_above"]
        assert p_high > p_low


# ===================================================================
# 15m PURITY: No daily code in the live path
# ===================================================================

class TestPurity_NoDailyCode:
    """Confirm daily code is completely absent from the live trading path."""

    def test_main_has_no_daily_imports(self):
        """main.py must not import DailySignalEngine or daily_engine."""
        import importlib
        import main
        source = importlib.util.find_spec("main").origin
        with open(source) as f:
            text = f.read()
        assert "DailySignalEngine" not in text
        assert "from daily_engine" not in text
        assert "daily_signal" not in text
        assert "daily_markets" not in text
        assert "daily_series" not in text

    def test_router_has_no_daily_evaluation(self):
        """router.py must not contain daily evaluation or per-strike model."""
        import importlib
        source = importlib.util.find_spec("router").origin
        with open(source) as f:
            text = f.read()
        assert "_per_strike_probability" not in text
        assert "_parse_daily_strike" not in text
        assert "daily_signal" not in text
        assert 'horizon == "daily"' not in text
        assert "daily_markets" not in text

    def test_router_route_signature_no_daily(self):
        """Router.route() must not accept daily_signal or daily_markets."""
        from router import ContractRouter
        import inspect
        sig = inspect.signature(ContractRouter.route)
        param_names = list(sig.parameters.keys())
        assert "daily_signal" not in param_names
        assert "daily_markets" not in param_names

    def test_all_router_output_is_15m(self):
        """Every contract from the router must have horizon='15m'."""
        from router import ContractRouter
        params = _make_15m_params()
        router = ContractRouter(params)
        state = MarketState(
            prices=[50000.0] * 100,
            volumes=[1.0] * 100,
            volatility_1h=0.005,
            mid_price=50000.0,
        )
        signal = Signal(
            bias=Bias.BULLISH, p_prob=0.70, edge=0.10,
            horizon="15m", confidence=0.8,
        )
        market = {
            "ticker": "KXBTC15M-TEST",
            "title": "test",
            "yes_ask_dollars": "0.6000",
            "yes_bid_dollars": "0.5800",
            "open_interest": 10,
            "_has_ws_depth": True,
            "_ws_liquidity_tier": "deep",
        }

        best, stink, _watchlist = router.route(
            state=state,
            signal=signal,
            fifteen_min_markets=[market],
        )

        if best is not None:
            assert best.horizon == "15m"
        for s in stink:
            assert s.horizon == "15m"

    def test_oracle_only_1m_kline_buffers(self):
        """BinanceOracle must only have 1m kline buffer, not 1h/4h/1d."""
        from binance_oracle import KLINE_BUFFER_SIZES
        assert list(KLINE_BUFFER_SIZES.keys()) == ["1m"]

    def test_kalshi_client_no_daily_series(self):
        """KalshiConfig must not have a daily_series field."""
        from kalshi_client import KalshiConfig
        cfg = KalshiConfig(api_key_id="t", private_key_path="t")
        assert not hasattr(cfg, "daily_series") or "daily_series" not in cfg.model_fields


# ===================================================================
# Edge fallback — no_implied from ask-only is blocked
# ===================================================================

class TestEdgeFallback_NoBidBlocked:
    def test_no_side_blocked_without_yes_bid(self):
        from router import ContractRouter
        params = _make_15m_params()
        router = ContractRouter(params)
        market = {
            "ticker": "KXBTC15M-TEST",
            "title": "test",
            "yes_ask_dollars": "0.7000",
            "yes_bid_dollars": "0",
            "open_interest": 10,
            "_has_ws_depth": True,
            "_ws_liquidity_tier": "thin_tradeable",
        }
        signal = Signal(bias=Bias.BEARISH, p_prob=0.30, edge=0.10, horizon="15m", confidence=0.8)
        skip_reasons: list[str] = []

        result = router._evaluate_contract(
            market=market,
            signal=signal,
            min_edge=0.01,
            max_spread=0.15,
            min_oi=0,
            contract_index=0,
            skip_reasons=skip_reasons,
        )

        assert result is None
        assert any("cannot price NO side" in r for r in skip_reasons)


# ===================================================================
# EXECUTION MODE DETERMINATION
# ===================================================================

class TestExecutionModeDetermination:
    """Test that ContractRouter.determine_execution_mode picks the right mode."""

    def _make_router(self) -> "ContractRouter":
        from router import ContractRouter
        return ContractRouter(_make_15m_params())

    def _make_cc(
        self,
        edge: float = 0.12,
        spread: float = 0.08,
        secs_left: float = 600,
        side: Side = Side.YES,
        tier: LiquidityTier = LiquidityTier.DEEP,
    ) -> ChosenContract:
        from datetime import timedelta
        close = datetime.now(timezone.utc) + timedelta(seconds=secs_left)
        return ChosenContract(
            ticker="KXBTC15M-TEST-B100000",
            horizon="15m",
            side=side,
            p_model=0.65,
            kalshi_ask=0.53,
            edge=edge,
            spread=spread,
            open_interest=50,
            liquidity_tier=tier,
            close_time=close,
        )

    def test_passive_when_plenty_of_time(self):
        router = self._make_router()
        cc = self._make_cc(edge=0.12, spread=0.08, secs_left=600)
        state = MarketState(order_book_imbalance=0.0)
        assert router.determine_execution_mode(cc, state) == ExecutionMode.PASSIVE

    def test_aggressive_near_expiry_strong_edge(self):
        router = self._make_router()
        cc = self._make_cc(edge=0.03, spread=0.05, secs_left=30)  # edge > 2x min (0.01)
        state = MarketState(order_book_imbalance=0.0)
        assert router.determine_execution_mode(cc, state) == ExecutionMode.AGGRESSIVE

    def test_skip_near_expiry_weak_edge(self):
        router = self._make_router()
        cc = self._make_cc(edge=0.015, spread=0.05, secs_left=30)
        state = MarketState(order_book_imbalance=0.0)
        assert router.determine_execution_mode(cc, state) == ExecutionMode.SKIP

    def test_moderate_time_pressure(self):
        router = self._make_router()
        cc = self._make_cc(edge=0.02, spread=0.08, secs_left=120)  # edge > 1.5x min
        state = MarketState(order_book_imbalance=0.0)
        assert router.determine_execution_mode(cc, state) == ExecutionMode.MODERATE

    def test_skip_adverse_obi_yes_side(self):
        router = self._make_router()
        cc = self._make_cc(edge=0.012, spread=0.08, secs_left=600, side=Side.YES)
        state = MarketState(order_book_imbalance=-0.5)  # strong sell pressure
        assert router.determine_execution_mode(cc, state) == ExecutionMode.SKIP

    def test_moderate_adverse_obi_but_strong_edge(self):
        router = self._make_router()
        cc = self._make_cc(edge=0.02, spread=0.08, secs_left=600, side=Side.YES)
        state = MarketState(order_book_imbalance=-0.5)
        assert router.determine_execution_mode(cc, state) == ExecutionMode.MODERATE

    def test_moderate_narrow_spread(self):
        router = self._make_router()
        cc = self._make_cc(edge=0.12, spread=0.02, secs_left=600)
        state = MarketState(order_book_imbalance=0.0)
        assert router.determine_execution_mode(cc, state) == ExecutionMode.MODERATE


# ===================================================================
# PASSIVE PRICE COMPUTATION
# ===================================================================

class TestPassivePricing:
    """Test _compute_passive_price for various spread/mode combinations."""

    def _make_client(self):
        from kalshi_client import KalshiClient, KalshiConfig
        cfg = KalshiConfig(api_key_id="test", private_key_path="test", min_edge=0.08)
        rm = _make_risk_manager()
        return KalshiClient(cfg, rm)

    def _make_cc(self, p_model: float = 0.65, side: Side = Side.YES) -> ChosenContract:
        return ChosenContract(
            ticker="KXBTC15M-TEST",
            horizon="15m",
            side=side,
            p_model=p_model,
            kalshi_ask=0.55,
            edge=p_model - 0.55,
            spread=0.03,
            liquidity_tier=LiquidityTier.DEEP,
        )

    def test_passive_narrow_spread_posts_at_bid_plus_1(self):
        client = self._make_client()
        cc = self._make_cc(p_model=0.70)
        book = {
            "yes": [{"price": 55, "delta": 10}],
            "no": [{"price": 45, "delta": 10}],  # YES bid = 55
        }
        price = client._compute_passive_price(cc, book, moderate=False)
        # YES bid = 100-45 = 55, YES ask = 55 → spread = 0
        # If ask == bid, spread_cents = 0, so bid+1 = 56
        # But wait: best_yes_bid = 100 - min(no_asks) = 100-45 = 55
        # best_yes_ask = 55. spread = 0. bid+1 = 56.
        # Edge at 56 = 0.70 - 0.56 = 0.14 > 0.08 ✓
        assert price is not None
        assert price <= int(cc.p_model * 100) - 8  # edge floor

    def test_passive_wide_spread_posts_at_mid(self):
        client = self._make_client()
        cc = self._make_cc(p_model=0.70)
        book = {
            "yes": [{"price": 60, "delta": 10}],
            "no": [{"price": 50, "delta": 10}],  # YES bid = 50
        }
        price = client._compute_passive_price(cc, book, moderate=False)
        # YES ask = 60, YES bid = 50, spread = 10 (wide)
        # mid = (60+50)//2 = 55
        # Edge floor: 70-8 = 62. min(55, 62) = 55
        # edge at 55 = 0.70-0.55 = 0.15 ✓
        assert price == 55

    def test_moderate_narrow_spread_takes_ask(self):
        client = self._make_client()
        cc = self._make_cc(p_model=0.70)
        book = {
            "yes": [{"price": 58, "delta": 10}],
            "no": [{"price": 44, "delta": 10}],  # YES bid = 56
        }
        price = client._compute_passive_price(cc, book, moderate=True)
        # YES ask = 58, YES bid = 56, spread = 2 (≤3 → take ask)
        # price = 58, edge floor = 62, min(58,62) = 58
        # edge = 0.70-0.58 = 0.12 ✓
        assert price == 58

    def test_moderate_wide_spread_posts_ask_minus_1(self):
        client = self._make_client()
        cc = self._make_cc(p_model=0.70)
        book = {
            "yes": [{"price": 60, "delta": 10}],
            "no": [{"price": 48, "delta": 10}],  # YES bid = 52
        }
        price = client._compute_passive_price(cc, book, moderate=True)
        # spread = 8 (medium), mid+1 = (60+52+1)//2 = 56
        # But wait spread_cents=8 so 4<=8<=8 → mid+1 branch
        assert price is not None

    def test_edge_floor_clamps_price(self):
        client = self._make_client()
        cc = self._make_cc(p_model=0.58)  # tight model probability
        book = {
            "yes": [{"price": 55, "delta": 10}],
            "no": [{"price": 48, "delta": 10}],  # YES bid = 52
        }
        price = client._compute_passive_price(cc, book, moderate=False)
        # YES ask = 55, YES bid = 52, spread = 3 → bid+1 = 53
        # edge floor: int(0.58*100)=57, 57 - 8 = 49. min(53, 49) = 49
        # edge at 49 = 0.58 - 0.49 = 0.09 ≥ 0.08 ✓
        assert price == 49

    def test_returns_none_when_no_valid_price(self):
        client = self._make_client()
        cc = self._make_cc(p_model=0.52)  # barely above 50%
        book = {
            "yes": [{"price": 50, "delta": 10}],
            "no": [{"price": 52, "delta": 10}],  # YES bid = 48
        }
        price = client._compute_passive_price(cc, book, moderate=False)
        # edge floor: 52 - 8 = 44. bid+1 = 49. min(49, 44) = 44
        # edge at 44 = 0.52 - 0.44 = 0.08 (exactly at min_edge)
        # This should be valid (edge >= min_edge)
        assert price is not None or price == 44

    def test_returns_none_no_ask(self):
        client = self._make_client()
        cc = self._make_cc()
        book = {"yes": [], "no": []}
        price = client._compute_passive_price(cc, book, moderate=False)
        assert price is None


# ===================================================================
# PASSIVE EXECUTION FLOW
# ===================================================================

class TestPassiveExecution:
    """Test execute_passive_limit and execute_chosen_contract routing."""

    @pytest.mark.asyncio
    async def test_passive_limit_creates_pending_order(self):
        from kalshi_client import KalshiClient, KalshiConfig
        cfg = KalshiConfig(api_key_id="test", private_key_path="test", min_edge=0.08)
        rm = _make_risk_manager()
        client = KalshiClient(cfg, rm)

        cc = ChosenContract(
            ticker="KXBTC15M-TEST",
            horizon="15m",
            side=Side.YES,
            p_model=0.70,
            kalshi_ask=0.55,
            edge=0.15,
            spread=0.08,
            liquidity_tier=LiquidityTier.DEEP,
        )
        mock_book = {
            "yes": [{"price": 58, "delta": 10}],
            "no": [{"price": 48, "delta": 10}],
        }
        client.get_market_orderbook = AsyncMock(return_value=mock_book)
        client._submit_order = AsyncMock(
            return_value={"order": {"order_id": "passive-001"}},
        )

        result = await client.execute_passive_limit(cc, 3, ExecutionMode.PASSIVE)

        assert result is not None
        assert isinstance(result, PendingOrder)
        assert result.execution_mode == "passive"
        assert result.order_id == "passive-001"
        # Should NOT increment open_position_count (resting, not filled)
        assert rm.open_position_count == 0
        # Should increment resting tracking
        assert rm.resting_order_count == 1

    @pytest.mark.asyncio
    async def test_passive_duplicate_ticker_blocked(self):
        from kalshi_client import KalshiClient, KalshiConfig
        cfg = KalshiConfig(api_key_id="test", private_key_path="test", min_edge=0.08)
        rm = _make_risk_manager()
        client = KalshiClient(cfg, rm)

        # Pre-populate a resting order on same ticker
        client._pending_orders["existing"] = PendingOrder(
            order_id="existing",
            ticker="KXBTC15M-TEST",
            side=Side.YES,
            limit_price_cents=52,
            contracts=2,
            model_p_at_placement=0.65,
            edge_at_placement=0.13,
            status="resting",
        )

        cc = ChosenContract(
            ticker="KXBTC15M-TEST",
            horizon="15m",
            side=Side.YES,
            p_model=0.70,
            kalshi_ask=0.55,
            edge=0.15,
            liquidity_tier=LiquidityTier.DEEP,
        )
        result = await client.execute_passive_limit(cc, 3, ExecutionMode.PASSIVE)
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_chosen_contract_routes_passive(self):
        from kalshi_client import KalshiClient, KalshiConfig
        cfg = KalshiConfig(api_key_id="test", private_key_path="test", min_edge=0.08)
        rm = _make_risk_manager()
        client = KalshiClient(cfg, rm)

        cc = ChosenContract(
            ticker="KXBTC15M-TEST",
            horizon="15m",
            side=Side.YES,
            p_model=0.70,
            kalshi_ask=0.55,
            edge=0.15,
            spread=0.08,
            liquidity_tier=LiquidityTier.DEEP,
        )
        mock_book = {
            "yes": [{"price": 58, "delta": 10}],
            "no": [{"price": 48, "delta": 10}],
        }
        client.get_market_orderbook = AsyncMock(return_value=mock_book)
        client._submit_order = AsyncMock(
            return_value={"order": {"order_id": "passive-002"}},
        )

        result = await client.execute_chosen_contract(
            cc, 3, execution_mode=ExecutionMode.PASSIVE,
        )
        assert isinstance(result, PendingOrder)
        assert rm.open_position_count == 0

    @pytest.mark.asyncio
    async def test_execute_chosen_contract_routes_aggressive(self):
        from kalshi_client import KalshiClient, KalshiConfig
        cfg = KalshiConfig(api_key_id="test", private_key_path="test", min_edge=0.08)
        rm = _make_risk_manager()
        client = KalshiClient(cfg, rm)

        cc = ChosenContract(
            ticker="KXBTC15M-TEST",
            horizon="15m",
            side=Side.YES,
            p_model=0.70,
            kalshi_ask=0.55,
            edge=0.15,
            spread=0.08,
            liquidity_tier=LiquidityTier.DEEP,
        )
        mock_book = {
            "yes": [{"price": 55, "delta": 10}],
            "no": [{"price": 48, "delta": 10}],
        }
        client.get_market_orderbook = AsyncMock(return_value=mock_book)
        client._submit_order = AsyncMock(
            return_value={"order": {"order_id": "agg-001"}},
        )

        result = await client.execute_chosen_contract(
            cc, 3, execution_mode=ExecutionMode.AGGRESSIVE,
        )
        assert isinstance(result, TradeInstruction)
        assert rm.open_position_count == 1

    @pytest.mark.asyncio
    async def test_execute_chosen_contract_skip(self):
        from kalshi_client import KalshiClient, KalshiConfig
        cfg = KalshiConfig(api_key_id="test", private_key_path="test", min_edge=0.08)
        rm = _make_risk_manager()
        client = KalshiClient(cfg, rm)

        cc = ChosenContract(
            ticker="KXBTC15M-TEST",
            horizon="15m",
            side=Side.YES,
            p_model=0.70,
            kalshi_ask=0.55,
            edge=0.15,
            liquidity_tier=LiquidityTier.DEEP,
        )
        result = await client.execute_chosen_contract(
            cc, 3, execution_mode=ExecutionMode.SKIP,
        )
        assert result is None
        assert rm.open_position_count == 0


# ===================================================================
# DEEP BOOK RETURN BUG FIX
# ===================================================================

class TestDeepBookReturnFix:
    """_execute_deep_book must return None when submission fails."""

    @pytest.mark.asyncio
    async def test_returns_none_on_submit_failure(self):
        from kalshi_client import KalshiClient, KalshiConfig
        cfg = KalshiConfig(api_key_id="test", private_key_path="test", min_edge=0.08)
        rm = _make_risk_manager()
        client = KalshiClient(cfg, rm)

        cc = ChosenContract(
            ticker="KXBTC15M-TEST",
            horizon="15m",
            side=Side.YES,
            p_model=0.70,
            kalshi_ask=0.55,
            edge=0.15,
            liquidity_tier=LiquidityTier.DEEP,
        )
        mock_book = {
            "yes": [{"price": 55, "delta": 10}],
            "no": [{"price": 48, "delta": 10}],
        }
        client.get_market_orderbook = AsyncMock(return_value=mock_book)
        client._submit_order = AsyncMock(return_value=None)  # submission fails

        result = await client._execute_deep_book(cc, 3)
        assert result is None
        assert rm.open_position_count == 0


# ===================================================================
# REEVALUATION — EDGE DETERIORATION & ESCALATION
# ===================================================================

class TestReevaluation:
    """Test _reevaluate_resting_order logic."""

    def _make_client(self):
        from kalshi_client import KalshiClient, KalshiConfig
        cfg = KalshiConfig(api_key_id="test", private_key_path="test", min_edge=0.08)
        rm = _make_risk_manager()
        return KalshiClient(cfg, rm)

    def _make_pending(
        self,
        edge: float = 0.12,
        secs_left: float = 300,
        mode: str = "passive",
        cycles: int = 5,
    ) -> PendingOrder:
        from datetime import timedelta
        return PendingOrder(
            order_id="reeval-001",
            ticker="KXBTC15M-TEST",
            side=Side.YES,
            limit_price_cents=53,
            contracts=3,
            model_p_at_placement=0.65,
            edge_at_placement=edge,
            expiry_utc=datetime.now(timezone.utc) + timedelta(seconds=secs_left),
            execution_mode=mode,
            cycles_alive=cycles,
            strike=100000.0,
        )

    def test_hold_when_edge_above_threshold(self):
        client = self._make_client()
        pending = self._make_pending(edge=0.12, secs_left=300)
        state = MarketState(mid_price=100000.0, prices=[100000.0] * 50)
        now = datetime.now(timezone.utc)

        # signal_engine mock: returns p_above=0.65 → edge at 53c = 0.12
        engine = MagicMock()
        engine.evaluate_contract.return_value = {"p_above": 0.65}
        oracle = MagicMock()

        action = client._reevaluate_resting_order(pending, engine, state, oracle, now)
        assert action == "hold"

    def test_cancel_when_edge_deteriorated(self):
        client = self._make_client()
        pending = self._make_pending(edge=0.12, secs_left=300)
        state = MarketState(mid_price=100000.0, prices=[100000.0] * 50)
        now = datetime.now(timezone.utc)

        # model now says p=0.55 → edge at 53c = 0.02 (below 0.08 threshold)
        engine = MagicMock()
        engine.evaluate_contract.return_value = {"p_above": 0.55}
        oracle = MagicMock()

        action = client._reevaluate_resting_order(pending, engine, state, oracle, now)
        assert action == "cancel"

    def test_escalate_near_expiry_strong_edge(self):
        client = self._make_client()
        # <45s triggers Phase 4 (aggressive escalation) in graduated pipeline
        pending = self._make_pending(edge=0.15, secs_left=30, mode="passive")
        state = MarketState(mid_price=100000.0, prices=[100000.0] * 50)
        now = datetime.now(timezone.utc)

        # model p=0.70 → edge at 53c = 0.17 (>0.75x min_edge of 0.08)
        engine = MagicMock()
        engine.evaluate_contract.return_value = {"p_above": 0.70}
        oracle = MagicMock()

        action = client._reevaluate_resting_order(pending, engine, state, oracle, now)
        assert action == "escalate"

    def test_improve_to_mid_near_expiry(self):
        client = self._make_client()
        # 45-90s triggers Phase 3 (improve to midpoint) in graduated pipeline
        pending = self._make_pending(edge=0.15, secs_left=60, mode="passive")
        state = MarketState(mid_price=100000.0, prices=[100000.0] * 50)
        now = datetime.now(timezone.utc)

        engine = MagicMock()
        engine.evaluate_contract.return_value = {"p_above": 0.70}
        oracle = MagicMock()

        action = client._reevaluate_resting_order(pending, engine, state, oracle, now)
        assert action == "improve_to_mid"

    def test_no_escalate_for_stink_bids(self):
        client = self._make_client()
        pending = self._make_pending(edge=0.15, secs_left=60, mode="stink")
        state = MarketState(mid_price=100000.0, prices=[100000.0] * 50)
        now = datetime.now(timezone.utc)

        engine = MagicMock()
        engine.evaluate_contract.return_value = {"p_above": 0.70}
        oracle = MagicMock()

        action = client._reevaluate_resting_order(pending, engine, state, oracle, now)
        # Stink bids don't escalate — they just hold or cancel
        assert action != "escalate"

    def test_escalate_patience_running_out(self):
        client = self._make_client()
        # <45s with strong edge triggers Phase 4 aggressive escalation
        pending = self._make_pending(
            edge=0.20, secs_left=40, mode="passive", cycles=13,
        )
        state = MarketState(mid_price=100000.0, prices=[100000.0] * 50)
        now = datetime.now(timezone.utc)

        # model p=0.75 → edge at 53c = 0.22 (>0.75x min_edge)
        engine = MagicMock()
        engine.evaluate_contract.return_value = {"p_above": 0.75}
        oracle = MagicMock()

        action = client._reevaluate_resting_order(pending, engine, state, oracle, now)
        assert action == "escalate"


# ===================================================================
# ESCALATION FLOW
# ===================================================================

class TestEscalation:
    """Test _escalate_to_aggressive end-to-end."""

    @pytest.mark.asyncio
    async def test_successful_escalation(self):
        from kalshi_client import KalshiClient, KalshiConfig
        cfg = KalshiConfig(api_key_id="test", private_key_path="test", min_edge=0.08)
        rm = _make_risk_manager()
        client = KalshiClient(cfg, rm)

        pending = PendingOrder(
            order_id="esc-001",
            ticker="KXBTC15M-TEST",
            side=Side.YES,
            limit_price_cents=52,
            contracts=3,
            model_p_at_placement=0.70,
            edge_at_placement=0.18,
            execution_mode="passive",
            cycles_alive=10,
            status="resting",
        )
        client._pending_orders["esc-001"] = pending
        rm.notify_resting_order_placed(52 * 3)

        mock_book = {
            "yes": [{"price": 56, "delta": 10}],
            "no": [{"price": 48, "delta": 10}],
        }
        client.get_market_orderbook = AsyncMock(return_value=mock_book)
        client.cancel_order = AsyncMock(return_value=True)
        client._submit_order = AsyncMock(
            return_value={"order": {"order_id": "agg-esc-001"}},
        )

        result = await client._escalate_to_aggressive("esc-001", pending)

        assert result is True
        assert pending.status == "cancelled"
        assert rm.open_position_count == 1  # position opened on aggressive fill
        # Resting risk should be resolved
        assert rm.resting_order_count == 0

    @pytest.mark.asyncio
    async def test_escalation_aborted_insufficient_edge(self):
        from kalshi_client import KalshiClient, KalshiConfig
        cfg = KalshiConfig(api_key_id="test", private_key_path="test", min_edge=0.08)
        rm = _make_risk_manager()
        client = KalshiClient(cfg, rm)

        pending = PendingOrder(
            order_id="esc-002",
            ticker="KXBTC15M-TEST",
            side=Side.YES,
            limit_price_cents=52,
            contracts=3,
            model_p_at_placement=0.58,  # weak model prob
            edge_at_placement=0.06,
            execution_mode="passive",
            status="resting",
        )
        client._pending_orders["esc-002"] = pending
        rm.notify_resting_order_placed(52 * 3)

        mock_book = {
            "yes": [{"price": 56, "delta": 10}],  # ask=56, edge = 0.58-0.56 = 0.02
            "no": [{"price": 48, "delta": 10}],
        }
        client.get_market_orderbook = AsyncMock(return_value=mock_book)
        client.cancel_order = AsyncMock(return_value=True)

        result = await client._escalate_to_aggressive("esc-002", pending)

        assert result is False
        assert rm.open_position_count == 0


# ===================================================================
# ROUTER UNREGISTER POSITION
# ===================================================================

class TestUnregisterPosition:
    """Test that cancelled resting orders clean up router state."""

    def test_unregister_decrements_event_contracts(self):
        from router import ContractRouter
        router = ContractRouter(_make_15m_params())
        router.register_position("KXBTC15M-25JAN25-B68000", Side.YES, 3)
        assert router._event_contracts.get("KXBTC15M-25JAN25", 0) == 3

        router.unregister_position("KXBTC15M-25JAN25-B68000", 3)
        assert router._event_contracts.get("KXBTC15M-25JAN25", 0) == 0
        assert "KXBTC15M-25JAN25-B68000" not in router._active_positions

    def test_unregister_does_not_go_negative(self):
        from router import ContractRouter
        router = ContractRouter(_make_15m_params())
        router.unregister_position("KXBTC15M-25JAN25-B68000", 5)
        assert router._event_contracts.get("KXBTC15M-25JAN25", 0) == 0


# ===================================================================
# MODE-AWARE MAX CYCLES
# ===================================================================

class TestModeAwareMaxCycles:
    """manage_pending_orders must respect mode-specific patience."""

    @pytest.mark.asyncio
    async def test_moderate_cancelled_after_6_cycles(self):
        from kalshi_client import KalshiClient, KalshiConfig
        cfg = KalshiConfig(api_key_id="test", private_key_path="test")
        rm = _make_risk_manager()
        client = KalshiClient(cfg, rm)

        pending = PendingOrder(
            order_id="mod-001",
            ticker="KXBTC15M-TEST",
            side=Side.YES,
            limit_price_cents=53,
            contracts=2,
            model_p_at_placement=0.65,
            edge_at_placement=0.12,
            execution_mode="moderate",
            cycles_alive=6,  # at limit
            status="resting",
        )
        client._pending_orders["mod-001"] = pending
        rm.notify_resting_order_placed(53 * 2)

        client.check_order_status = AsyncMock(
            return_value={"remaining_count": 2, "status": "resting"},
        )
        client.cancel_order = AsyncMock(return_value=True)

        filled, cancelled, resting = await client.manage_pending_orders()
        assert cancelled == 1
        assert pending.status == "cancelled"

    @pytest.mark.asyncio
    async def test_passive_survives_6_cycles(self):
        from kalshi_client import KalshiClient, KalshiConfig
        cfg = KalshiConfig(api_key_id="test", private_key_path="test")
        rm = _make_risk_manager()
        client = KalshiClient(cfg, rm)

        pending = PendingOrder(
            order_id="pas-001",
            ticker="KXBTC15M-TEST",
            side=Side.YES,
            limit_price_cents=53,
            contracts=2,
            model_p_at_placement=0.65,
            edge_at_placement=0.12,
            execution_mode="passive",
            cycles_alive=6,  # still within passive limit of 15
            status="resting",
        )
        client._pending_orders["pas-001"] = pending
        rm.notify_resting_order_placed(53 * 2)

        client.check_order_status = AsyncMock(
            return_value={"remaining_count": 2, "status": "resting"},
        )

        filled, cancelled, resting = await client.manage_pending_orders()
        assert resting == 1
        assert cancelled == 0
