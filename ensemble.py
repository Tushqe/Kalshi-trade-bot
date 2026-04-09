"""
Ensemble Engine.

Aggregates outputs from N strategies into a single probability estimate
using a weighted consensus formula and sigmoid mapping.

Consensus formula
-----------------
    S = Σ (w_i · c_i · s_i)

where:
    w_i = static weight for strategy i
    c_i = self-reported confidence of signal i
    s_i = directional score of signal i  ∈ [-1, 1]

Probability mapping
-------------------
    p = σ(k · S) = 1 / (1 + exp(-k · S))

where k is a temperature parameter controlling sigmoid steepness.
"""

from __future__ import annotations

import logging
import math

from models import EnsemblePrediction, StrategySignal

logger = logging.getLogger(__name__)

# Default static weights (must sum to 1.0)
DEFAULT_WEIGHTS: dict[str, float] = {
    "trend_following": 0.30,
    "mean_reversion": 0.20,
    "microstructure": 0.25,
    "regime_volatility": 0.25,
}

DEFAULT_SIGMOID_K: float = 3.0  # steepness — higher = more extreme probabilities


class EnsembleEngine:
    """
    Combines multiple ``StrategySignal`` objects into an ``EnsemblePrediction``.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        sigmoid_k: float = DEFAULT_SIGMOID_K,
    ) -> None:
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.sigmoid_k = sigmoid_k

        # Validate weights sum ≈ 1
        w_sum = sum(self.weights.values())
        if abs(w_sum - 1.0) > 0.01:
            logger.warning("Strategy weights sum to %.3f (expected 1.0) — normalising", w_sum)
            self.weights = {k: v / w_sum for k, v in self.weights.items()}

    def combine(self, signals: list[StrategySignal]) -> EnsemblePrediction:
        """
        Weighted consensus → sigmoid → base probability.

        Strategies not present in ``self.weights`` are silently skipped.
        If no signals carry weight, returns a neutral 0.50 prediction.
        """
        weighted_sum = 0.0
        active_weight = 0.0

        for sig in signals:
            w = self.weights.get(sig.strategy_name, 0.0)
            if w == 0.0:
                logger.debug("Signal from '%s' has no weight — skipping", sig.strategy_name)
                continue
            contribution = w * sig.confidence * sig.score
            weighted_sum += contribution
            active_weight += w
            logger.debug(
                "Ensemble | %s  w=%.2f c=%.3f s=%.3f -> contrib=%.4f",
                sig.strategy_name, w, sig.confidence, sig.score, contribution,
            )

        # Re-normalise if some strategies were absent
        if active_weight > 0 and active_weight < 1.0:
            weighted_sum /= active_weight

        base_prob = self._sigmoid(weighted_sum)

        logger.info(
            "Ensemble result | S=%.4f p=%.4f (k=%.1f, %d signals, active_w=%.2f)",
            weighted_sum, base_prob, self.sigmoid_k, len(signals), active_weight,
        )

        return EnsemblePrediction(
            base_probability=base_prob,
            raw_score=weighted_sum,
            component_signals=signals,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sigmoid(self, x: float) -> float:
        """
        Numerically stable sigmoid: σ(k·x).

        For large positive kx use 1 / (1 + exp(-kx));
        for large negative kx use exp(kx) / (1 + exp(kx)) to avoid overflow.
        """
        z = self.sigmoid_k * x
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)
