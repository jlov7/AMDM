import pytest

from eval.offline_eval import DetectorRecord, _latency_metrics


def test_latency_metrics_computes_rates_and_latency():
    records = [
        DetectorRecord(turn_id=0, label=False, detected=False, score=0.0, agent_id="a"),
        DetectorRecord(turn_id=1, label=True, detected=False, score=2.0, agent_id="a"),
        DetectorRecord(turn_id=2, label=True, detected=True, score=3.5, agent_id="a"),
        DetectorRecord(turn_id=3, label=False, detected=True, score=1.0, agent_id="a"),
        DetectorRecord(turn_id=0, label=False, detected=False, score=0.0, agent_id="b"),
        DetectorRecord(turn_id=1, label=True, detected=False, score=2.5, agent_id="b"),
        DetectorRecord(turn_id=2, label=True, detected=False, score=2.6, agent_id="b"),
        DetectorRecord(turn_id=3, label=True, detected=True, score=3.0, agent_id="b"),
        DetectorRecord(turn_id=4, label=False, detected=False, score=0.2, agent_id="b"),
    ]
    metrics = _latency_metrics(records)
    assert metrics["false_positive_rate"] == pytest.approx(0.25)
    assert metrics["detection_rate"] == pytest.approx(1.0)
    assert metrics["latency_mean"] == pytest.approx(1.5)
