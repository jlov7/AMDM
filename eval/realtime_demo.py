"""Realtime AMDM demo: stream trace events and emit anomalies."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import pandas as pd
from rich.console import Console
from rich.table import Table

from amdm.amdm import AMDMConfig, AMDMMonitor
from amdm.detectors import (
    CostLatencySpikeDetector,
    DetectorEvent,
    GoalDriftDetector,
    ToolErrorBurstDetector,
)
from amdm.features import FeatureExtractor
from amdm.ingestion import parse_normalized_line, guess_source_type
from amdm.metrics import MetricsSink, OTelJSONSink, PrometheusMetricsSink


logger = logging.getLogger("amdm.realtime_demo")
console = Console()

TURN_PAYLOAD_SCHEMA = {
    "type": "object",
    "properties": {
        "agent_id": {"type": "string"},
        "turn_id": {"type": "integer"},
        "score": {"type": "number"},
        "threshold": {"type": "number"},
        "is_alerting": {"type": "boolean"},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "detector": {"type": "string"},
                    "turn_id": {"type": "integer"},
                    "score": {"type": "number"},
                    "threshold": {"type": "number"},
                    "severity": {"type": "string"},
                    "message": {"type": "string"},
                    "agent_id": {"type": "string"},
                },
                "required": ["detector", "turn_id", "score", "threshold", "severity", "message", "agent_id"],
            },
        },
        "features": {"type": "object"},
        "raw": {"type": "object"},
    },
    "required": ["agent_id", "turn_id", "score", "threshold", "is_alerting", "events"],
}

FEATURE_WEIGHTS = {
    "tokens_in": 0.6,
    "tokens_out": 0.6,
    "tokens_total": 0.9,
    "token_io_ratio": 0.7,
    "latency_ms": 1.2,
    "latency_per_token": 1.3,
    "approval_count": 0.8,
    "tool_error_rate": 1.1,
    "tool_error_density": 1.2,
    "goal_to_plan_delta": 1.4,
    "plan_to_action_delta": 1.4,
    "loop_depth": 0.5,
    "cost_usd": 0.9,
}

def _detect_source(path: Path):
    if path.suffix in {".jsonl", ".json"} and "otel" in path.name:
        return "otel_genai"
    return "agents_sdk"


def _build_monitor(feature_columns: List[str]) -> AMDMMonitor:
    weights = [FEATURE_WEIGHTS.get(name, 1.0) for name in feature_columns]
    config = AMDMConfig(
        alpha=0.18,
        cov_window=80,
        min_baseline=12,
        alert_threshold=2.4,
        hysteresis=0.4,
        feature_weights=weights,
        distance_quantile=0.85,
    )
    return AMDMMonitor(feature_columns, config)


def run_demo(args: argparse.Namespace):
    if args.stdin and args.follow:
        logger.warning("--follow is ignored when reading from stdin.")
        args.follow = False

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    extractor = FeatureExtractor()
    monitor_by_agent: Dict[str, AMDMMonitor] = {}
    detectors_by_agent: Dict[str, List] = {}

    sink = open(args.output, "w", encoding="utf-8") if args.output else None

    metric_sinks: List[MetricsSink] = []
    if not args.no_metrics:
        if args.prometheus_port is not None:
            try:
                metric_sinks.append(PrometheusMetricsSink(port=args.prometheus_port))
            except ImportError:  # pragma: no cover - handled during runtime only
                logger.warning("prometheus_client not installed; skipping Prometheus sink")
        if args.otel_metrics is not None:
            metric_sinks.append(OTelJSONSink(args.otel_metrics))

    source = args.source or (None if args.stdin else _detect_source(args.input))
    feature_columns = extractor.FEATURE_COLUMNS
    turn_counters: Dict[str, int] = defaultdict(int)
    state_interval = args.state_interval if args.state_interval and args.state_interval > 0 else 1
    events_since_persist = 0

    state_path = args.state
    if state_path is not None:
        state_path = Path(state_path)
        if state_path.exists():
            try:
                saved_state = json.loads(state_path.read_text())
                saved_features = saved_state.get("feature_columns")
                if saved_features and saved_features != feature_columns:
                    logger.warning("Feature column mismatch in saved state; using current extractor columns.")
                for agent_id, payload in saved_state.get("monitors", {}).items():
                    monitor_by_agent[agent_id] = AMDMMonitor.from_dict(payload)
                    detectors_by_agent[agent_id] = [
                        GoalDriftDetector(),
                        ToolErrorBurstDetector(),
                        CostLatencySpikeDetector(),
                    ]
                for agent_id, counter in saved_state.get("turn_counters", {}).items():
                    turn_counters[agent_id] = int(counter)
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.error("Failed to load state file: %s", exc)
                state_path = None

    def auto_parse_line(line: str) -> dict | None:
        nonlocal source
        candidates = []
        if source:
            candidates.append(source)
        guessed = guess_source_type(line)
        if guessed and guessed not in candidates:
            candidates.append(guessed)
        for fallback in ("agents_sdk", "otel_genai"):
            if fallback not in candidates:
                candidates.append(fallback)
        last_error = None
        for candidate in candidates:
            try:
                record = parse_normalized_line(line, candidate)
                source = candidate
                return record
            except Exception as exc:  # pragma: no cover - runtime guard
                last_error = exc
                if source:
                    break
                continue
        if last_error is not None:
            logger.error("Failed to parse line: %s", last_error)
        return None

    def persist_state(force: bool = False):
        nonlocal events_since_persist
        if state_path is None:
            return
        if not force and events_since_persist < state_interval:
            return
        state_payload = {
            "feature_columns": feature_columns,
            "monitors": {
                agent: monitor.to_dict()
                for agent, monitor in monitor_by_agent.items()
            },
            "turn_counters": turn_counters,
        }
        tmp_path = state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
        tmp_path.replace(state_path)
        events_since_persist = 0

    def handle_record(record: dict) -> None:
        nonlocal events_since_persist
        if record is None:
            return

        event_frame = pd.DataFrame([record])
        feature_set = extractor.transform(event_frame)
        row = feature_set.frame.iloc[0].to_dict()

        agent_id = row.get("agent_id") or row.get("trace_id") or "agent"
        if agent_id not in monitor_by_agent:
            monitor_by_agent[agent_id] = _build_monitor(feature_columns)
            detectors_by_agent[agent_id] = [
                GoalDriftDetector(),
                ToolErrorBurstDetector(),
                CostLatencySpikeDetector(),
            ]
        monitor = monitor_by_agent[agent_id]
        detectors = detectors_by_agent[agent_id]

        turn_id = row.get("turn_id")
        if turn_id is None or (isinstance(turn_id, float) and pd.isna(turn_id)):
            turn_id = turn_counters[agent_id]
        turn_id = int(turn_id)
        row["turn_id"] = turn_id
        turn_counters[agent_id] = max(turn_counters[agent_id], turn_id + 1)

        features = [row.get(col, 0.0) for col in feature_columns]
        state = monitor.update(features)
        row["amdm_score"] = state.last_score
        row["amdm_threshold"] = state.threshold

        if metric_sinks:
            for metrics_sink in metric_sinks:
                metrics_sink.observe_turn(agent_id, state.last_score, state.threshold, state.is_alerting)

        events_emitted: List[DetectorEvent] = []
        emitted_payloads: List[dict] = []
        for detector in detectors:
            event = detector.update(row)
            if not event:
                continue
            events_emitted.append(event)
            payload = {
                "detector": event.detector,
                "turn_id": event.turn_id,
                "score": event.score,
                "threshold": state.threshold,
                "severity": event.severity,
                "message": event.message,
                "agent_id": agent_id,
            }
            emitted_payloads.append(payload)
            if metric_sinks:
                for metrics_sink in metric_sinks:
                    metrics_sink.observe_event(agent_id, event.detector, event.severity, event.score)

        turn_payload = {
            "agent_id": agent_id,
            "turn_id": turn_id,
            "score": state.last_score,
            "threshold": state.threshold,
            "is_alerting": state.is_alerting,
            "events": emitted_payloads,
        }
        if args.include_features:
            turn_payload["features"] = {col: row.get(col, 0.0) for col in feature_columns}
        if args.include_raw:
            raw_copy = record.copy()
            if args.redact_raw:
                for field in ("goal_text", "plan_text", "action_text"):
                    if field in raw_copy:
                        raw_copy[field] = "[REDACTED]"
            turn_payload["raw"] = {
                key: (value.isoformat() if isinstance(value, pd.Timestamp) else value)
                for key, value in raw_copy.items()
            }

        if sink:
            sink.write(json.dumps(turn_payload) + "\n")
            sink.flush()

        event_names = [payload["detector"] for payload in emitted_payloads]

        if args.summary or args.summary_only:
            summary = (
                f"[summary] agent={agent_id} turn={turn_id} score={state.last_score:.2f} "
                f"events={event_names}"
            )
            print(summary, file=sys.stderr)

        emit_json = args.json_only and not args.summary_only
        emit_table = not args.json_only and not args.summary_only

        if emit_json:
            serialized = json.dumps(turn_payload, indent=2 if args.json_pretty else None)
            print(serialized)
        if emit_table:
            event_name = ", ".join(ev.detector for ev in events_emitted) or "-"
            detail = events_emitted[0].message if events_emitted else ""
            table = Table(title="AMDM Realtime Demo")
            table.add_column("Turn")
            table.add_column("Agent")
            table.add_column("Score")
            table.add_column("Threshold")
            table.add_column("Event")
            table.add_column("Detail")
            table.add_row(
                str(turn_id),
                agent_id,
                f"{state.last_score:.2f}",
                f"{state.threshold:.2f}",
                event_name,
                detail,
            )
            console.print(table)

        time.sleep(args.sleep)
        events_since_persist += 1
        persist_state()

    if args.stdin:
        stream = sys.stdin
        for line in stream:
            record = auto_parse_line(line)
            handle_record(record)
    else:
        with args.input.open("r", encoding="utf-8") as stream:
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line:
                    if not args.follow:
                        break
                    stream.seek(position)
                    time.sleep(args.poll_interval)
                    continue
                record = auto_parse_line(line)
                handle_record(record)

    if sink:
        sink.close()
    persist_state(force=True)
    for metrics_sink in metric_sinks:
        metrics_sink.close()


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime AMDM demo.")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("examples/sample_traces/agents_trace.jsonl"),
        help="Path to JSONL trace stream.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write anomaly events (JSONL).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep interval between turns to emulate streaming.",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Continuously poll the input file for new events.",
    )
    parser.add_argument(
        "--poll-interval",
        dest="poll_interval",
        type=float,
        default=1.0,
        help="Polling interval (seconds) when --follow is enabled.",
    )
    parser.add_argument(
        "--prometheus-port",
        type=int,
        default=None,
        help="Expose Prometheus metrics on the given port.",
    )
    parser.add_argument(
        "--otel-metrics",
        type=Path,
        default=None,
        help="Write OTEL-friendly NDJSON metrics to the given path.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=None,
        help="Persist AMDM monitor state to this JSON file (load if it exists).",
    )
    parser.add_argument(
        "--state-interval",
        type=int,
        default=1,
        help="Persist state every N events (default 1).",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read trace lines from stdin instead of a file.",
    )
    parser.add_argument(
        "--source",
        choices=["agents_sdk", "otel_genai"],
        default=None,
        help="Explicitly set the trace source type (required when using --stdin).",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Suppress table output and emit JSON events only.",
    )
    parser.add_argument(
        "--json-pretty",
        action="store_true",
        help="Pretty-print JSON output when --json-only is set.",
    )
    parser.add_argument(
        "--include-features",
        action="store_true",
        help="Include normalized feature values in JSON output when --json-only is set.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Include raw normalized event payload in JSON output.",
    )
    parser.add_argument(
        "--redact-raw",
        action="store_true",
        help="Redact goal/plan/action fields when including raw payloads.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Emit a one-line text summary per turn alongside JSON/table output.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Emit only the textual summary per turn (implies --summary).",
    )
    parser.add_argument(
        "--json-schema",
        action="store_true",
        help="Print the JSON schema for per-turn payloads and exit.",
    )
    parser.add_argument(
        "--log-level",
        default="warning",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Set logging verbosity for runtime diagnostics.",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable Prometheus/OTEL metric sinks.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None):
    args = parse_args(argv or sys.argv[1:])
    if args.summary_only:
        args.summary = True
        args.json_only = False
    if args.json_schema:
        print(json.dumps(TURN_PAYLOAD_SCHEMA, indent=2))
        return
    run_demo(args)


if __name__ == "__main__":
    main()
