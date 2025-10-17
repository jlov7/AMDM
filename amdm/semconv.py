"""OpenTelemetry GenAI semantic convention helpers."""

from __future__ import annotations

OTEL_GENAI_PREFIX = "genai."

# Common semantic attribute keys for GenAI spans.
SEM_ATTRIBUTES = {
    "input_tokens": f"{OTEL_GENAI_PREFIX}usage.input_tokens",
    "output_tokens": f"{OTEL_GENAI_PREFIX}usage.output_tokens",
    "model_latency_ms": f"{OTEL_GENAI_PREFIX}runtime.latency",
    "approval_state": f"{OTEL_GENAI_PREFIX}response.approval_state",
    "tool_calls": f"{OTEL_GENAI_PREFIX}response.tool_calls",
    "tool_errors": f"{OTEL_GENAI_PREFIX}response.tool_errors",
    "loop_depth": f"{OTEL_GENAI_PREFIX}session.loop_depth",
    "goal_text": f"{OTEL_GENAI_PREFIX}session.goal",
    "plan_text": f"{OTEL_GENAI_PREFIX}session.plan",
    "action_text": f"{OTEL_GENAI_PREFIX}session.action",
    "cost_usd": f"{OTEL_GENAI_PREFIX}usage.cost",
}


def attribute_get(attributes: dict, key: str, default=None):
    """Safe getter for attribute dictionaries."""
    if not isinstance(attributes, dict):
        return default
    return attributes.get(key, default)
