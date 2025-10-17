from pathlib import Path

import pandas as pd

from amdm.ingestion import (
    AgentsSDKIngestor,
    OTelGenAIIngestor,
    parse_agents_line,
    parse_otel_line,
    guess_source_type,
)


DATA_DIR = Path("examples/sample_traces")


def test_agents_sdk_ingestion_normalizes_fields():
    ingestor = AgentsSDKIngestor()
    result = ingestor.load(DATA_DIR / "agents_trace.jsonl")
    frame = result.events
    assert not frame.empty
    assert {"tokens_in", "tokens_out", "latency_ms", "tool_error_rate"}.issubset(frame.columns)
    assert frame["tool_error_rate"].iloc[-1] > 0


def test_otel_genai_ingestion_normalizes_fields():
    ingestor = OTelGenAIIngestor()
    result = ingestor.load(DATA_DIR / "otel_trace.otel.jsonl")
    frame = result.events
    assert not frame.empty
    assert frame["agent_id"].iloc[0] == "agent-otel-1"
    assert frame["tool_error_rate"].iloc[1] > 0


def test_parse_agents_line_matches_batch_ingestion():
    lines = (DATA_DIR / "agents_trace.jsonl").read_text().splitlines()
    record = parse_agents_line(lines[0])
    ingestor = AgentsSDKIngestor()
    batch_record = ingestor.load(DATA_DIR / "agents_trace.jsonl").events.iloc[0].to_dict()
    assert record["agent_id"] == batch_record["agent_id"]
    assert record["tokens_out"] == batch_record["tokens_out"]
    assert record["tool_error_rate"] == batch_record["tool_error_rate"]


def test_parse_otel_line_matches_batch_ingestion():
    lines = (DATA_DIR / "otel_trace.otel.jsonl").read_text().splitlines()
    record = parse_otel_line(lines[0])
    ingestor = OTelGenAIIngestor()
    batch_record = ingestor.load(DATA_DIR / "otel_trace.otel.jsonl").events.iloc[0].to_dict()
    assert record["agent_id"] == batch_record["agent_id"]
    assert record["tokens_in"] == batch_record["tokens_in"]
    assert record["tool_error_rate"] == batch_record["tool_error_rate"]


def test_guess_source_type_detects_formats():
    agents_line = (DATA_DIR / "agents_trace.jsonl").read_text().splitlines()[0]
    otel_line = (DATA_DIR / "otel_trace.otel.jsonl").read_text().splitlines()[0]
    assert guess_source_type(agents_line) == "agents_sdk"
    assert guess_source_type(otel_line) == "otel_genai"
    assert guess_source_type("invalid json") is None
