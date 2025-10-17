"""Metric sinks for exposing AMDM telemetry to external systems."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional


class MetricsSink:
    """Interface for collecting AMDM metrics."""

    def observe_turn(self, agent_id: str, score: float, threshold: float, is_alerting: bool) -> None:
        raise NotImplementedError

    def observe_event(self, agent_id: str, detector: str, severity: str, score: float) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class PrometheusMetricsSink(MetricsSink):
    """Expose gauges and counters via the Prometheus client library."""

    def __init__(self, port: int = 9109):
        from prometheus_client import Counter, Gauge, start_http_server

        self._score_gauge = Gauge("amdm_score", "Latest AMDM score per agent.", ["agent_id"])
        self._threshold_gauge = Gauge("amdm_threshold", "Adaptive AMDM threshold per agent.", ["agent_id"])
        self._alert_gauge = Gauge("amdm_alert_active", "Whether AMDM is in alerting state.", ["agent_id"])
        self._event_counter = Counter(
            "amdm_detector_events_total",
            "Count of detector events emitted.",
            ["agent_id", "detector", "severity"],
        )
        start_http_server(port)

    def observe_turn(self, agent_id: str, score: float, threshold: float, is_alerting: bool) -> None:
        self._score_gauge.labels(agent_id=agent_id).set(score)
        self._threshold_gauge.labels(agent_id=agent_id).set(threshold)
        self._alert_gauge.labels(agent_id=agent_id).set(1.0 if is_alerting else 0.0)

    def observe_event(self, agent_id: str, detector: str, severity: str, score: float) -> None:
        self._event_counter.labels(agent_id=agent_id, detector=detector, severity=severity).inc()


class OTelJSONSink(MetricsSink):
    """Write AMDM metrics to NDJSON with OpenTelemetry-friendly fields."""

    def __init__(self, path: Path):
        self._fh = Path(path).open("w", encoding="utf-8")

    def observe_turn(self, agent_id: str, score: float, threshold: float, is_alerting: bool) -> None:
        payload = {
            "resource": {"amdm.agent_id": agent_id},
            "metric": "amdm.score",
            "score": score,
            "threshold": threshold,
            "is_alerting": is_alerting,
        }
        self._fh.write(json.dumps(payload) + "\n")
        self._fh.flush()

    def observe_event(self, agent_id: str, detector: str, severity: str, score: float) -> None:
        payload = {
            "resource": {"amdm.agent_id": agent_id},
            "metric": "amdm.detector_event",
            "detector": detector,
            "severity": severity,
            "score": score,
        }
        self._fh.write(json.dumps(payload) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
