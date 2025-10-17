# Real-time JSON Payload

Every turn processed by `python -m eval.realtime_demo` can be exported as a JSON object. The schema matches the value returned by `--json-schema`.

```jsonc
{
  "agent_id": "agent-1",            // string identifier (Agents SDK session or OTel span attr)
  "turn_id": 42,                     // integer turn sequence (auto-generated if missing)
  "score": 3.91,                     // latest AMDM Mahalanobis score
  "threshold": 2.40,                 // adaptive threshold in effect for this turn
  "is_alerting": true,               // hysteresis state (true when score is above alert threshold)
  "events": [                        // zero or more detector events for this turn
    {
      "detector": "goal_drift",
      "turn_id": 42,
      "score": 3.91,
      "threshold": 2.40,
      "severity": "high",
      "message": "Goal drift suspected…",
      "agent_id": "agent-1"
    }
  ],
  "features": {                      // optional (requires --include-features)
    "tokens_in": 140,
    "tokens_out": 250,
    "latency_ms": 1100,
    "tool_error_rate": 0.25,
    "goal_to_plan_delta": 0.80,
    "plan_to_action_delta": 0.86,
    "loop_depth": 1,
    "cost_usd": 0.016,
    "token_io_ratio": 1.78,
    "latency_per_token": 2.82,
    "tool_error_density": 0.0016
  },
  "raw": {                           // optional (requires --include-raw)
    "trace_id": "trace-1",
    "span_id": "span-3",
    "timestamp": "2024-06-01T12:00:10+00:00",
    "tokens_in": 140,
    "tokens_out": 250,
    "latency_ms": 1100,
    "approval_count": 1,
    "tool_error_rate": 0.25,
    "tool_calls": 4,
    "tool_errors": 1,
    "goal_text": "Gather intel",
    "plan_text": "Search -> Summarize",
    "action_text": "Tool retry",
    "loop_depth": 1,
    "cost_usd": 0.016,
    "source": "agents_sdk"
  }
}
```

## Optional redaction

`--redact-raw` masks the `goal_text`, `plan_text`, and `action_text` fields before serialising. Use it when raw payloads may contain sensitive information.

## Summary lines

`--summary` (or `--summary-only`) prints a concise text line to stderr:

```
[summary] agent=agent-1 turn=42 score=3.91 events=['goal_drift']
```

Redirect stdout/stderr independently to integrate JSON output with logging pipelines, e.g.:

```bash
python -m eval.realtime_demo traces/run.jsonl \
  --json-only --summary 1>turns.jsonl 2>turns.log
```
