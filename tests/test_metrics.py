from pathlib import Path
import json

import pytest

from amdm.metrics import OTelJSONSink


def test_otel_json_sink_writes_metrics(tmp_path: Path):
    output = tmp_path / "metrics.jsonl"
    sink = OTelJSONSink(output)
    sink.observe_turn("agent-1", 2.5, 2.0, True)
    sink.observe_event("agent-1", "goal_drift", "high", 3.1)
    sink.close()

    lines = output.read_text().strip().splitlines()
    assert len(lines) == 2
    turn_payload = json.loads(lines[0])
    event_payload = json.loads(lines[1])
    assert turn_payload["resource"]["amdm.agent_id"] == "agent-1"
    assert pytest.approx(turn_payload["score"]) == 2.5
    assert event_payload["detector"] == "goal_drift"
