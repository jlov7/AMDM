from amdm.detectors import (
    CostLatencySpikeDetector,
    GoalDriftDetector,
    ToolErrorBurstDetector,
)


def test_goal_drift_detector_requires_consecutive_turns():
    detector = GoalDriftDetector(distance_threshold=2.0, adherence_threshold=0.3, consecutive=2, threshold_multiplier=1.0, min_approval=0.5)
    turn = {"turn_id": 1, "amdm_score": 2.5, "amdm_threshold": 2.0, "plan_to_action_delta": 0.4, "goal_to_plan_delta": 0.5, "approval_count": 0.0}
    assert detector.update(turn) is None
    turn["turn_id"] = 2
    event = detector.update(turn)
    assert event is not None
    assert event.detector == "goal_drift"


def test_tool_error_burst_detector_emits_after_band_crossing():
    detector = ToolErrorBurstDetector(
        ewma_alpha=0.5,
        band_multiplier=1.0,
        min_burst=2,
        min_rate_floor=0.0,
        dynamic_sigma=0.0,
    )
    normal_turn = {"turn_id": 0, "tool_error_rate": 0.0, "tool_errors": 0.0, "amdm_score": 0.5}
    detector.update(normal_turn)
    turn_high = {"turn_id": 1, "tool_error_rate": 0.9, "tool_errors": 2.0, "amdm_score": 3.0}
    assert detector.update(turn_high) is None
    turn_high["turn_id"] = 2
    event = detector.update(turn_high)
    assert event is not None
    assert event.detector == "tool_error_burst"


def test_tool_error_burst_detector_handles_multiple_bursts():
    detector = ToolErrorBurstDetector(
        ewma_alpha=0.4,
        band_multiplier=1.0,
        min_burst=1,
        min_rate_floor=0.0,
        dynamic_sigma=0.3,
        sustain_turns=0,
        freeze_during_burst=True,
    )
    for turn_id in range(5):
        detector.update({"turn_id": turn_id, "tool_error_rate": 0.02, "tool_errors": 0.0, "amdm_score": 0.2})
    event1 = detector.update({"turn_id": 10, "tool_error_rate": 1.0, "tool_errors": 1.0, "amdm_score": 3.0})
    assert event1 is not None
    for turn_id in range(11, 16):
        detector.update({"turn_id": turn_id, "tool_error_rate": 0.05, "tool_errors": 0.0, "amdm_score": 0.5})
    event2 = detector.update({"turn_id": 20, "tool_error_rate": 0.9, "tool_errors": 1.0, "amdm_score": 3.0})
    assert event2 is not None
    assert event2.detector == "tool_error_burst"


def test_cost_latency_spike_detector_flags_latency():
    detector = CostLatencySpikeDetector(distance_threshold=2.0, latency_threshold_ms=2000, threshold_multiplier=1.0)
    turn = {
        "turn_id": 3,
        "amdm_score": 2.5,
        "amdm_threshold": 2.0,
        "latency_ms": 5000,
        "tokens_in": 1000,
        "tokens_out": 1000,
        "latency_per_token": 5.0,
    }
    event = detector.update(turn)
    assert event is not None
    assert event.detector == "cost_latency_spike"
