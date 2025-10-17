# Adaptive Multi-Dimensional Monitoring (AMDM)

Adaptive Multi-Dimensional Monitoring (AMDM) is a research-focused toolkit for detecting emergent anomalies in agentic systems. It ingests OpenAI Agents SDK traces and generic OpenTelemetry (OTel) GenAI spans, normalizes them into comparable turn-level events, derives behavioral features, and scores those features with an adaptive EWMA + Mahalanobis distance model. Detectors consume those scores to flag goal drift, tool error bursts, and cost/latency spikes in near-real-time.

## Features
- Ingestion adapters for Agents SDK JSON exports and OTel GenAI NDJSON
- Feature extraction capturing tokens, latency, approval signals, tool errors, and plan/action deltas
- Adaptive Multi-Dimensional Monitoring core with weighted per-axis EWMA normalization, joint Mahalanobis distance, rolling covariance baseline, and quantile-driven hysteresis thresholds
- Streaming detectors for key anomaly classes plus offline evaluation and reporting
- Synthetic scenario generator for stress-testing detection quality, complete with labeled anomalies
- Real-time demo that incrementally tails trace or stdin streams, optionally follows new events, and emits anomaly events with AMDM scores and thresholds

## Quickstart
1. Create a virtual environment with Python 3.10+
2. Install in editable mode:
   ```bash
   pip install -e .
   ```
3. Run the unit tests:
   ```bash
   pytest -q
   ```
4. Execute the offline evaluation harness:
   ```bash
   python -m eval.offline_eval
   ```
5. Launch the real-time demo against sample traces:
   ```bash
   python -m eval.realtime_demo examples/sample_traces/agents_trace.jsonl
   # Follow a growing file with 0.5s polling
   python -m eval.realtime_demo examples/sample_traces/agents_trace.jsonl --follow --poll-interval 0.5
   # Expose Prometheus metrics and write OTEL NDJSON alongside
   python -m eval.realtime_demo examples/sample_traces/agents_trace.jsonl --prometheus-port 9109 --otel-metrics eval/metrics.jsonl
# Resume from persisted AMDM state (see below)
python -m eval.realtime_demo examples/sample_traces/agents_trace.jsonl --state state.json
# Stream from stdin (source is auto-detected)
   cat examples/sample_traces/agents_trace.jsonl | python -m eval.realtime_demo --stdin --json-only
   ```

## Trace Export Guidance
### OpenAI Agents SDK
Use the SDK's trace export utilities to dump session traces to JSONL. The ingestion module expects each line to contain a turn-level record with token counts, latency, tool invocations, and approval metadata. See `examples/sample_traces/agents_trace.jsonl` for a minimal example.

### OTel GenAI
Export GenAI spans via OTLP -> JSONL using `otlp-exporter` or a collector pipeline. Ensure spans include the GenAI semantic attributes defined in the [OTel semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). The parser in `amdm.ingestion` reads NDJSON where each line is a span with `traceId`, `spanId`, `attributes`, and timestamps.

## Documentation
- `notebooks/AMDM_explainer.md` outlines the algorithmic intuition and references the AMDM literature.
- `eval/offline_eval.py` generates `eval_report.md` with overall and per-scenario metrics plus ROC/PR curves.
- `eval/realtime_demo.py` streams anomalies with a stable AMDM score, adaptive threshold, optional follow mode, Prometheus/OTel metric sinks, and evaluation latencies/false-positive rates in the generated report.

Flags of note for `eval/realtime_demo.py`:
- `--json-only` emits per-turn NDJSON (with `events`, optional `features`, and `raw` payloads); layer `--json-pretty`, `--include-features`, `--include-raw`, and `--summary`/`--summary-only` for richer pipelines.
- `--state` / `--state-interval` persist monitor baselines without replaying the entire stream.
- `--no-metrics` disables Prometheus/OTEL sinks when you only need CLI output.
- `--json-schema` prints the turn payload schema and exits.

## Intended Use
This repository targets research and operational experimentation. It is not optimized for commercial deployment and comes with no service-level guarantees.

## License
Distributed under the Apache License 2.0. See `LICENSE` for details.

## Persisting Monitor State
The realtime demo supports persisting AMDM monitor state so restarts can resume without discarding baselines:

```bash
   python -m eval.realtime_demo examples/sample_traces/agents_trace.jsonl --state state.json --state-interval 10 --json-only
   ```

The CLI loads existing state from `state.json` (if present) at startup and writes updated state on each new event (or every `--state-interval` events). State includes per-agent EWMA means/variances, covariance matrices, and alert status.

Use `--json-only` to emit turn-level NDJSON objects (one per turn with a list of detector events) to stdout; omit `--output` to skip writing a JSONL file.
