"""Detection primitives built atop AMDM scores and feature deltas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DetectorEvent:
    detector: str
    turn_id: int
    score: float
    severity: str
    message: str


class BaseDetector:
    name: str = "base"

    def update(self, turn: Dict) -> Optional[DetectorEvent]:
        raise NotImplementedError


class GoalDriftDetector(BaseDetector):
    name = "goal_drift"

    def __init__(
        self,
        distance_threshold: float = 2.4,
        adherence_threshold: float = 0.75,
        consecutive: int = 1,
        threshold_multiplier: float = 1.02,
        min_approval: float = 0.5,
        sustain_turns: int = 1,
    ):
        self.distance_threshold = distance_threshold
        self.adherence_threshold = adherence_threshold
        self.consecutive = consecutive
        self.threshold_multiplier = threshold_multiplier
        self.min_approval = min_approval
        self.sustain_turns = sustain_turns
        self._streak = 0
        self._active_window = 0

    def update(self, turn: Dict) -> Optional[DetectorEvent]:
        score = turn.get("amdm_score", 0.0)
        delta = turn.get("plan_to_action_delta", 0.0)
        turn_id = int(turn.get("turn_id", 0))
        approval = float(turn.get("approval_count", 0.0) or 0.0)
        goal_delta = turn.get("goal_to_plan_delta", 0.0)
        threshold = max(
            self.distance_threshold,
            float(turn.get("amdm_threshold", 0.0) or 0.0),
        )
        effective_threshold = threshold * self.threshold_multiplier
        if (
            score >= effective_threshold
            and delta >= self.adherence_threshold
            and goal_delta >= 0.4
            and approval <= self.min_approval
        ):
            self._streak += 1
            if self._streak >= self.consecutive:
                message = (
                    f"Goal drift suspected: AMDM score {score:.2f}, "
                    f"plan-action delta {delta:.2f} over {self._streak} turns."
                )
                self._streak = max(self.consecutive - 1, 0)
                self._active_window = self.sustain_turns
                return DetectorEvent(
                    detector=self.name,
                    turn_id=turn_id,
                    score=score,
                    severity="high",
                    message=message,
                )
        else:
            self._streak = 0
            if delta < self.adherence_threshold:
                self._active_window = 0

        if (
            self._active_window > 0
            and delta >= self.adherence_threshold
            and approval <= self.min_approval
        ):
            self._active_window -= 1
            return DetectorEvent(
                detector=self.name,
                turn_id=turn_id,
                score=score,
                severity="medium",
                message=(
                    f"Goal drift persisting: AMDM score {score:.2f}, "
                    f"plan-action delta {delta:.2f}."
                ),
            )

        return None


class ToolErrorBurstDetector(BaseDetector):
    name = "tool_error_burst"

    def __init__(
        self,
        ewma_alpha: float = 0.3,
        band_multiplier: float = 1.5,
        min_burst: int = 2,
        min_rate_floor: float = 0.05,
        dynamic_sigma: float = 1.0,
        sustain_turns: int = 2,
        freeze_during_burst: bool = True,
    ):
        self.ewma_alpha = ewma_alpha
        self.band_multiplier = band_multiplier
        self.min_burst = min_burst
        self.min_rate_floor = min_rate_floor
        self.dynamic_sigma = dynamic_sigma
        self.sustain_turns = sustain_turns
        self.freeze_during_burst = freeze_during_burst
        self._ewma = 0.0
        self._ewma_sq = 0.0
        self._streak = 0
        self._baseline_count = 0
        self._rate_mean = 0.0
        self._rate_m2 = 0.0
        self._active_burst = 0

    def update(self, turn: Dict) -> Optional[DetectorEvent]:
        rate = float(turn.get("tool_error_rate", 0.0))
        score = float(turn.get("amdm_score", 0.0))
        tool_errors = float(turn.get("tool_errors", 0.0) or 0.0)
        turn_id = int(turn.get("turn_id", 0))

        # Baseline stats prior to observing current rate.
        if self._baseline_count > 1:
            variance_baseline = self._rate_m2 / (self._baseline_count - 1)
            std_baseline = max(variance_baseline, 0.0) ** 0.5
        else:
            std_baseline = 0.0
        mean_baseline = self._rate_mean if self._baseline_count else 0.0
        dynamic_min_rate = max(self.min_rate_floor, mean_baseline + self.dynamic_sigma * std_baseline)

        variance = max(self._ewma_sq - self._ewma**2, 0.0)
        std = variance**0.5
        upper_band = self._ewma + self.band_multiplier * std

        threshold = max(upper_band, dynamic_min_rate)

        event_obj: Optional[DetectorEvent] = None

        if rate >= threshold and tool_errors > 0:
            self._streak += 1
            if self._streak >= self.min_burst:
                message = (
                    f"Tool error burst: rate {rate:.2f} exceeds threshold {threshold:.2f} "
                    f"for {self._streak} turns."
                )
                self._streak = 0
                self._active_burst = self.sustain_turns
                event_obj = DetectorEvent(
                    detector=self.name,
                    turn_id=turn_id,
                    score=score,
                    severity="medium",
                    message=message,
                )
        else:
            self._streak = 0

        freeze_current = self.freeze_during_burst and (rate >= threshold or self._active_burst > 0)

        if event_obj is None:
            if self._active_burst > 0 and rate >= self.min_rate_floor and tool_errors > 0:
                self._active_burst -= 1
                event_obj = DetectorEvent(
                    detector=self.name,
                    turn_id=turn_id,
                    score=score,
                    severity="medium",
                    message=(
                        f"Tool error burst continuing: rate {rate:.2f} above baseline "
                        f"threshold {threshold:.2f}."
                    ),
                )
            else:
                if self._active_burst > 0:
                    self._active_burst = max(self._active_burst - 1, 0)

        if not freeze_current:
            self._ewma = (1 - self.ewma_alpha) * self._ewma + self.ewma_alpha * rate
            self._ewma_sq = (1 - self.ewma_alpha) * self._ewma_sq + self.ewma_alpha * (rate**2)
        else:
            # Minimal decay during freeze to avoid stagnation.
            self._ewma *= 0.95
            self._ewma_sq *= 0.95

        if not freeze_current:
            self._baseline_count, self._rate_mean, self._rate_m2 = self._welford_update(
                self._baseline_count, self._rate_mean, self._rate_m2, rate
            )
        return event_obj

    @staticmethod
    def _welford_update(count: int, mean: float, m2: float, value: float) -> tuple[int, float, float]:
        count += 1
        delta = value - mean
        mean += delta / count if count else value
        m2 += delta * (value - mean)
        return count, mean, m2


class CostLatencySpikeDetector(BaseDetector):
    name = "cost_latency_spike"

    def __init__(
        self,
        distance_threshold: float = 2.8,
        latency_threshold_ms: float = 4500.0,
        latency_per_token_threshold: float = 8.0,
        threshold_multiplier: float = 1.05,
        latency_sigma: float = 2.0,
        token_sigma: float = 2.0,
        latency_per_token_sigma: float = 1.5,
        token_threshold_total: float = 2 * 4096,
    ):
        self.distance_threshold = distance_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.latency_per_token_threshold = latency_per_token_threshold
        self.threshold_multiplier = threshold_multiplier
        self.latency_sigma = latency_sigma
        self.token_sigma = token_sigma
        self.latency_per_token_sigma = latency_per_token_sigma
        self.token_threshold_total = token_threshold_total
        self._lat_count = 0
        self._lat_mean = 0.0
        self._lat_m2 = 0.0
        self._token_count = 0
        self._token_mean = 0.0
        self._token_m2 = 0.0
        self._lpt_count = 0
        self._lpt_mean = 0.0
        self._lpt_m2 = 0.0

    def update(self, turn: Dict) -> Optional[DetectorEvent]:
        score = float(turn.get("amdm_score", 0.0))
        latency = float(turn.get("latency_ms", 0.0))
        tokens = float(turn.get("tokens_in", 0.0) + turn.get("tokens_out", 0.0))
        latency_per_token = float(turn.get("latency_per_token", 0.0) or 0.0)
        turn_id = int(turn.get("turn_id", 0))
        threshold = max(
            self.distance_threshold,
            float(turn.get("amdm_threshold", 0.0) or 0.0),
        ) * self.threshold_multiplier

        lat_mean, lat_std = self._baseline_stats(
            self._lat_count, self._lat_mean, self._lat_m2, default_value=latency
        )
        token_mean, token_std = self._baseline_stats(
            self._token_count, self._token_mean, self._token_m2, default_value=tokens
        )
        lpt_mean, lpt_std = self._baseline_stats(
            self._lpt_count, self._lpt_mean, self._lpt_m2, default_value=latency_per_token
        )

        dynamic_latency_threshold = max(
            self.latency_threshold_ms, lat_mean + self.latency_sigma * lat_std
        )
        dynamic_token_threshold = max(
            self.token_threshold_total, token_mean + self.token_sigma * token_std
        )
        dynamic_lpt_threshold = max(
            self.latency_per_token_threshold,
            lpt_mean + self.latency_per_token_sigma * lpt_std,
        )

        high_cost = (
            latency >= dynamic_latency_threshold
            or tokens >= dynamic_token_threshold
            or latency_per_token >= dynamic_lpt_threshold
        )

        if score >= threshold and high_cost:
            message = (
                f"Cost/latency spike: score {score:.2f}, latency {latency:.1f} ms, "
                f"tokens {tokens:.0f}, latency/token {latency_per_token:.2f}."
            )
            self._update_baselines(latency, tokens, latency_per_token)
            return DetectorEvent(
                detector=self.name,
                turn_id=turn_id,
                score=score,
                severity="medium",
                message=message,
            )
        self._update_baselines(latency, tokens, latency_per_token)
        return None

    @staticmethod
    def _baseline_stats(count: int, mean: float, m2: float, default_value: float) -> tuple[float, float]:
        if count <= 1:
            return default_value, 0.0
        variance = m2 / (count - 1)
        return mean, max(variance, 0.0) ** 0.5

    def _update_baselines(self, latency: float, tokens: float, latency_per_token: float) -> None:
        self._lat_count, self._lat_mean, self._lat_m2 = self._welford_update(
            self._lat_count, self._lat_mean, self._lat_m2, latency
        )
        self._token_count, self._token_mean, self._token_m2 = self._welford_update(
            self._token_count, self._token_mean, self._token_m2, tokens
        )
        self._lpt_count, self._lpt_mean, self._lpt_m2 = self._welford_update(
            self._lpt_count, self._lpt_mean, self._lpt_m2, latency_per_token
        )

    @staticmethod
    def _welford_update(count: int, mean: float, m2: float, value: float) -> tuple[int, float, float]:
        count += 1
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)
        return count, mean, m2
