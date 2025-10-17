"""Adaptive Multi-Dimensional Monitoring core implementation."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Deque, Dict, List, Optional
from collections import deque

import numpy as np


@dataclass
class AMDMConfig:
    alpha: float = 0.2
    cov_window: int = 50
    min_baseline: int = 10
    hysteresis: float = 0.5
    alert_threshold: float = 3.0
    recover_threshold: Optional[float] = None
    jitter: float = 1e-3
    feature_weights: Optional[List[float]] = None
    distance_quantile: Optional[float] = None


@dataclass
class AMDMState:
    last_score: float = 0.0
    is_alerting: bool = False
    mean_vector: np.ndarray | None = None
    cov_matrix: np.ndarray | None = None
    inverse_cov: np.ndarray | None = None
    threshold: float = 0.0


class AMDMMonitor:
    """Compute AMDM scores via per-axis EWMA and Mahalanobis distance."""

    def __init__(self, feature_names: List[str], config: Optional[AMDMConfig] = None):
        self.feature_names = feature_names
        self.config = config or AMDMConfig()
        self._means = np.zeros(len(feature_names))
        self._variances = np.ones(len(feature_names))
        self._baseline: Deque[np.ndarray] = deque(maxlen=self.config.cov_window)
        self.state = AMDMState()
        self._initialized = False
        self._weights = np.ones(len(feature_names))
        if self.config.feature_weights is not None:
            weights = np.asarray(self.config.feature_weights, dtype=float)
            if weights.shape[0] != len(feature_names):
                raise ValueError("feature_weights length must match feature_names length")
            self._weights = weights
        self._distance_history: Deque[float] = deque(maxlen=self.config.cov_window)
        self._current_threshold = self.config.alert_threshold

    def reset(self):
        self._means = np.zeros(len(self.feature_names))
        self._variances = np.ones(len(self.feature_names))
        self._baseline.clear()
        self.state = AMDMState()
        self._initialized = False
        self._distance_history.clear()
        self._current_threshold = self.config.alert_threshold

    def update(self, values: List[float] | np.ndarray) -> AMDMState:
        vector = np.asarray(values, dtype=float)
        if vector.shape[0] != len(self.feature_names):
            raise ValueError("Feature vector length mismatch.")

        if not self._initialized:
            self._means = vector.copy()
            self._variances = np.ones_like(vector)
            self._initialized = True

        alpha = self.config.alpha
        delta = vector - self._means
        self._means += alpha * delta
        self._variances = (1 - alpha) * self._variances + alpha * (delta**2)
        std = np.sqrt(np.maximum(self._variances, self.config.jitter))

        zscores = np.divide(
            vector - self._means,
            std,
            out=np.zeros_like(vector),
            where=std > 0,
        )
        weighted = zscores * self._weights

        self._baseline.append(weighted)
        mean_vec, cov_matrix, inv_cov = self._compute_covariance()
        score = self._mahalanobis(weighted, mean_vec, inv_cov)
        threshold = self._update_threshold(score)

        alert = self._apply_hysteresis(score, threshold)
        self.state = AMDMState(
            last_score=score,
            is_alerting=alert,
            mean_vector=mean_vec,
            cov_matrix=cov_matrix,
            inverse_cov=inv_cov,
            threshold=threshold,
        )
        return self.state

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "config": asdict(self.config),
            "means": self._means.tolist(),
            "variances": self._variances.tolist(),
            "baseline": [vec.tolist() for vec in self._baseline],
            "distance_history": list(self._distance_history),
            "current_threshold": self._current_threshold,
            "state": {
                "last_score": self.state.last_score,
                "is_alerting": self.state.is_alerting,
                "threshold": self.state.threshold,
            },
            "initialized": self._initialized,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "AMDMMonitor":
        feature_names = payload["feature_names"]
        config = AMDMConfig(**payload["config"])
        monitor = cls(feature_names, config)
        monitor._means = np.array(payload["means"], dtype=float)
        monitor._variances = np.array(payload["variances"], dtype=float)
        monitor._baseline.clear()
        for vec in payload.get("baseline", []):
            monitor._baseline.append(np.array(vec, dtype=float))
        monitor._distance_history.clear()
        for value in payload.get("distance_history", []):
            monitor._distance_history.append(float(value))
        monitor._current_threshold = float(payload.get("current_threshold", config.alert_threshold))
        state_payload = payload.get("state", {})
        monitor.state = AMDMState(
            last_score=state_payload.get("last_score", 0.0),
            is_alerting=state_payload.get("is_alerting", False),
            threshold=state_payload.get("threshold", config.alert_threshold),
        )
        monitor._initialized = payload.get("initialized", False)
        return monitor

    def _compute_covariance(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(self._baseline) < max(self.config.min_baseline, 2):
            dim = len(self.feature_names)
            identity = np.eye(dim)
            return np.zeros(dim), identity, identity
        data = np.vstack(self._baseline)
        mean_vec = data.mean(axis=0)
        cov_matrix = np.cov(data.T, bias=False)
        cov_matrix += np.eye(cov_matrix.shape[0]) * self.config.jitter
        inv_cov = np.linalg.pinv(cov_matrix)
        return mean_vec, cov_matrix, inv_cov

    def _mahalanobis(self, zscores: np.ndarray, mean_vec: np.ndarray, inv_cov: np.ndarray) -> float:
        diff = zscores - mean_vec
        score = float(np.sqrt(diff.T @ inv_cov @ diff))
        return score

    def _update_threshold(self, score: float) -> float:
        self._distance_history.append(score)
        if (
            self.config.distance_quantile is not None
            and len(self._distance_history) >= self.config.min_baseline
        ):
            candidate = float(
                np.quantile(
                    np.array(self._distance_history, dtype=float),
                    self.config.distance_quantile,
                )
            )
            self._current_threshold = max(self.config.alert_threshold, candidate)
        return self._current_threshold

    def _apply_hysteresis(self, score: float, threshold: float) -> bool:
        recover = self.config.recover_threshold or (threshold - self.config.hysteresis)
        recover = max(recover, 0.0)
        if self.state.is_alerting:
            if score <= recover:
                return False
            return True
        return score >= threshold
