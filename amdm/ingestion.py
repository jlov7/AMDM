"""Trace ingestion for Agents SDK and OTel GenAI exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import pandas as pd

from . import semconv


@dataclass
class IngestionResult:
    """Normalized turn-level events."""

    events: pd.DataFrame
    source: str


def _parse_timestamp(value: str | int | float | None) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return pd.to_datetime(value, unit="s", utc=True)
    try:
        return pd.Timestamp(datetime.fromisoformat(value))
    except ValueError:
        return None


def _default_turn_record() -> dict:
    return {
        "trace_id": None,
        "span_id": None,
        "agent_id": None,
        "turn_id": None,
        "timestamp": None,
        "tokens_in": 0.0,
        "tokens_out": 0.0,
        "latency_ms": 0.0,
        "approval_count": 0.0,
        "tool_error_rate": 0.0,
        "tool_calls": 0.0,
        "tool_errors": 0.0,
        "goal_text": "",
        "plan_text": "",
        "action_text": "",
        "loop_depth": 0.0,
        "cost_usd": 0.0,
    }


def _estimate_tool_error_rate(tool_calls: float, tool_errors: float) -> float:
    if tool_calls <= 0:
        return 0.0
    return float(tool_errors) / float(tool_calls)


def _normalize_agents_payload(payload: dict) -> dict:
    record = _default_turn_record()
    record["trace_id"] = payload.get("trace_id")
    record["span_id"] = payload.get("span_id")
    record["agent_id"] = payload.get("agent_id")
    record["turn_id"] = payload.get("turn_id")
    record["timestamp"] = _parse_timestamp(payload.get("timestamp"))

    metrics = payload.get("metrics", {})
    record["tokens_in"] = metrics.get("tokens_in", 0.0)
    record["tokens_out"] = metrics.get("tokens_out", 0.0)
    record["latency_ms"] = metrics.get("latency_ms", 0.0)
    record["approval_count"] = payload.get("approvals", 0)

    tool_calls = payload.get("tool_invocations", {})
    record["tool_calls"] = tool_calls.get("total", 0.0)
    record["tool_errors"] = tool_calls.get("errors", 0.0)
    record["tool_error_rate"] = _estimate_tool_error_rate(
        record["tool_calls"], record["tool_errors"]
    )

    session = payload.get("session", {})
    record["goal_text"] = session.get("goal", "")
    record["plan_text"] = session.get("plan", "")
    record["action_text"] = session.get("action", "")
    record["loop_depth"] = session.get("loop_depth", 0.0)
    record["cost_usd"] = metrics.get("cost_usd", 0.0)
    return record


def _normalize_otel_payload(payload: dict) -> dict:
    attributes = payload.get("attributes", {})
    record = _default_turn_record()

    record["trace_id"] = payload.get("traceId") or payload.get("trace_id")
    record["span_id"] = payload.get("spanId") or payload.get("span_id")
    record["agent_id"] = attributes.get("agent.id")
    record["turn_id"] = attributes.get("agent.turn_id")

    start_time = payload.get("startTimeUnixNano")
    end_time = payload.get("endTimeUnixNano")
    if start_time:
        record["timestamp"] = pd.to_datetime(int(start_time), unit="ns", utc=True)
    else:
        record["timestamp"] = _parse_timestamp(payload.get("timestamp"))
    if start_time and end_time:
        latency_ns = int(end_time) - int(start_time)
        record["latency_ms"] = latency_ns / 1_000_000.0

    record["tokens_in"] = semconv.attribute_get(
        attributes, semconv.SEM_ATTRIBUTES["input_tokens"], 0.0
    )
    record["tokens_out"] = semconv.attribute_get(
        attributes, semconv.SEM_ATTRIBUTES["output_tokens"], 0.0
    )
    latency_attr = semconv.attribute_get(
        attributes, semconv.SEM_ATTRIBUTES["model_latency_ms"]
    )
    if latency_attr is not None:
        record["latency_ms"] = latency_attr

    approval_state = semconv.attribute_get(
        attributes, semconv.SEM_ATTRIBUTES["approval_state"]
    )
    if approval_state == "approved":
        record["approval_count"] = 1.0

    tool_calls = semconv.attribute_get(
        attributes, semconv.SEM_ATTRIBUTES["tool_calls"], 0.0
    )
    tool_errors = semconv.attribute_get(
        attributes, semconv.SEM_ATTRIBUTES["tool_errors"], 0.0
    )
    record["tool_calls"] = float(tool_calls or 0.0)
    record["tool_errors"] = float(tool_errors or 0.0)
    record["tool_error_rate"] = _estimate_tool_error_rate(
        record["tool_calls"], record["tool_errors"]
    )

    record["loop_depth"] = semconv.attribute_get(
        attributes, semconv.SEM_ATTRIBUTES["loop_depth"], 0.0
    )
    record["goal_text"] = semconv.attribute_get(
        attributes, semconv.SEM_ATTRIBUTES["goal_text"], ""
    ) or ""
    record["plan_text"] = semconv.attribute_get(
        attributes, semconv.SEM_ATTRIBUTES["plan_text"], ""
    ) or ""
    record["action_text"] = semconv.attribute_get(
        attributes, semconv.SEM_ATTRIBUTES["action_text"], ""
    ) or ""
    record["cost_usd"] = semconv.attribute_get(
        attributes, semconv.SEM_ATTRIBUTES["cost_usd"], 0.0
    )
    return record


def parse_agents_line(line: str) -> dict | None:
    text = line.strip()
    if not text:
        return None
    payload = json.loads(text)
    record = _normalize_agents_payload(payload)
    record["source"] = "agents_sdk"
    record["timestamp"] = pd.to_datetime(record["timestamp"], utc=True)
    return record


def parse_otel_line(line: str) -> dict | None:
    text = line.strip()
    if not text:
        return None
    payload = json.loads(text)
    record = _normalize_otel_payload(payload)
    record["source"] = "otel_genai"
    record["timestamp"] = pd.to_datetime(record["timestamp"], utc=True)
    return record


def parse_normalized_line(line: str, source: str) -> dict | None:
    if source == "agents_sdk":
        return parse_agents_line(line)
    if source == "otel_genai":
        return parse_otel_line(line)
    raise ValueError(f"Unsupported source type: {source}")


def guess_source_type(line: str) -> Optional[str]:
    """Best-effort guess of the source type for a JSON trace line."""

    text = line.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        if "attributes" in payload or "startTimeUnixNano" in payload:
            return "otel_genai"
        if "metrics" in payload or "tool_invocations" in payload:
            return "agents_sdk"
    return None


class AgentsSDKIngestor:
    """Normalize traces exported from the OpenAI Agents SDK."""

    def load(self, path: str | Path) -> IngestionResult:
        path = Path(path)
        records: List[dict] = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            record = _normalize_agents_payload(payload)
            records.append(record)

        frame = pd.DataFrame(records)
        frame["source"] = "agents_sdk"
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame.sort_values(["agent_id", "turn_id"], inplace=True, ignore_index=True)
        return IngestionResult(events=frame, source="agents_sdk")


class OTelGenAIIngestor:
    """Normalize OTLP -> JSONL exports following the GenAI semantic conventions."""

    def load(self, path: str | Path) -> IngestionResult:
        path = Path(path)
        records: List[dict] = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            record = _normalize_otel_payload(payload)
            records.append(record)

        frame = pd.DataFrame(records)
        frame["source"] = "otel_genai"
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame.sort_values(["agent_id", "turn_id"], inplace=True, ignore_index=True)
        return IngestionResult(events=frame, source="otel_genai")


def ingest_paths(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Auto-detect file types and concatenate normalized events."""

    frames: List[pd.DataFrame] = []
    for path in paths:
        path = Path(path)
        if path.name.endswith(".otel.jsonl"):
            result = OTelGenAIIngestor().load(path)
        else:
            result = AgentsSDKIngestor().load(path)
        frames.append(result.events)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)
