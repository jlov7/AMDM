# Adaptive Multi-Dimensional Monitoring (AMDM)

Adaptive Multi-Dimensional Monitoring is an end-to-end toolkit for observing agentic systems, detecting emergent anomalies in near-real time, and benchmarking detector performance against synthetic scenarios. It unifies ingestion of OpenAI Agents SDK traces and OpenTelemetry (OTel) GenAI spans, derives rich behavioral features, and scores each turn with an adaptive EWMA + Mahalanobis distance model. Detector modules consume the scoring stream to flag goal drift, tool error bursts, and cost/latency spikes the moment they appear.

---

## Why AMDM matters

| Challenge | AMDM contribution |
|-----------|-------------------|
| **Heterogeneous telemetry** across Agents SDK and OTel pipelines | Normalizes both sources into a single turn-level schema (see [JSON Payload](docs/JSON_PAYLOAD.md)) |
| **Drift in agent behavior** as plans or goals evolve | Adaptive per-axis EWMA with rolling covariance keeps pace with regime changes while surfacing real anomalies |
| **Operational triage** requires actionable categories | Detector stack classifies anomalies into goal drift, tool error bursts, and cost/latency spikes with configurable thresholds |
| **Continuous improvement** demands reproducible evaluation | Simulator + offline harness generate labeled scenarios with ROC/PR plots, precision/recall/F1, latency, and false-positive statistics |
| **Streaming monitoring** must integrate with existing pipelines | Real-time CLI tails files or stdin, persists state, emits NDJSON per turn, and exposes Prometheus/OTel metrics for dashboards |

AMDM is ideal for research groups and operations teams that need to benchmark or monitor agent behaviours without building a full stack from scratch.

---

## Repository tour

| Path | Purpose |
|------|---------|
| `amdm/` | Core library: ingestion adapters, feature engineering, AMDM monitor, detectors, labeling utilities |
| `sim/` | Scenario generator and configurations for producing labeled synthetic traces |
| `eval/` | Offline evaluation harness (`offline_eval.py`) and real-time demo CLI (`realtime_demo.py`) |
| `examples/` | Sample Agents SDK & OTel traces for smoke-testing and tutorials |
| `tests/` | Pytest suite covering math, ingestion, detectors, metrics | 
| `docs/` | Additional references (CLI guide, JSON payload schema) |
| `notebooks/` | Narrative explainer of the AMDM algorithm |

---

## Quickstart

```bash
# 1. Create an isolated Python environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate

# 2. Install AMDM in editable mode
pip install -e .

# 3. Run the test suite
pytest -q

# 4. Benchmark detectors against synthetic scenarios
python -m eval.offline_eval

# 5. Stream real-time detections from the sample trace
python -m eval.realtime_demo examples/sample_traces/agents_trace.jsonl --json-only --summary
```

Offline evaluation produces `eval/eval_report.md`, `eval/metrics.csv`, and ROC/PR plots, while the real-time demo prints per-turn JSON (or an interactive table) so you can watch anomalies unfold.

---

## Trace ingestion

### Agents SDK exports

Export a session to JSONL using the OpenAI Agents SDK. Each line should include turn-level metadata (tokens, latencies, tool invocations, approvals, session goal/plan/action). See `examples/sample_traces/agents_trace.jsonl`.

```bash
python -m eval.realtime_demo path/to/agents_trace.jsonl --summary --json-only
```

### OTel GenAI spans

Pipe OTLP exports (e.g., `otelcol` -> JSONL) that implement the [OTel GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). The parser reads `traceId`, `spanId`, `attributes`, and timestamps from each span. Example: `examples/sample_traces/otel_trace.otel.jsonl`.

```bash
cat otel_trace.otel.jsonl | python -m eval.realtime_demo --stdin --json-only
```

`amdm.ingestion.guess_source_type` auto-detects the source format, so stdin streams require no additional flags.

---

## AMDM pipeline at a glance

```
           ┌────────────┐      ┌──────────────┐      ┌────────────┐
Trace ───▶ │ Ingestion  │ ───▶ │ Feature      │ ───▶ │ AMDM        │ ───▶ Detectors & Alerts
           │ (Agents/   │      │ Extraction   │      │ Monitor     │      (goal drift, tool errors,
           │  OTel)     │      │ (tokens,     │      │ (EWMA +     │       cost/latency spikes)
           └────────────┘      │ latencies,   │      │ Mahalanobis)│
                               │ approvals…)  │      └────────────┘
                               └──────────────┘

Optional loops:
  • Simulator generates synthetic traces with labeled anomalies
  • Offline evaluation computes metrics, latency, ROC/PR plots
  • Real-time demo streams detections, persists state, exports metrics
```

---

## Using the offline evaluation harness

```bash
python -m eval.offline_eval
```

Outputs:

- `eval/eval_report.md` – Markdown summary with precision/recall/F1, ROC AUC, PR AUC, false positive rate, and detection latency per detector
- `eval/metrics.csv` – Tabular data suitable for spreadsheets
- `eval/scenario_metrics.csv` – Per-scenario breakdown when applicable
- `eval/roc_*.png`, `eval/pr_*.png` – ROC/PR curves for goal drift, tool error burst, and cost/latency spike detectors

Use these artifacts to benchmark configuration changes before promoting a detector into production.

---

## Real-time demo overview

Run `python -m eval.realtime_demo --help` for the complete option set. The most commonly used combinations are summarised below—full details live in [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md).

### Tail a trace file interactively

```bash
python -m eval.realtime_demo traces/run.jsonl --follow --poll-interval 1.0
```

### Emit per-turn NDJSON to stdout (machine-friendly)

```bash
python -m eval.realtime_demo traces/run.jsonl \
  --json-only \
  --include-features \
  --include-raw --redact-raw \
  --summary --state state.json --state-interval 10
```

This prints a summary line to **stderr** and a JSON object per turn to **stdout**. Raw payloads are redacted for goal/plan/action text. The same payload structure is documented [here](docs/JSON_PAYLOAD.md). To inspect the schema programmatically:

```bash
python -m eval.realtime_demo --json-schema
```

### Pipe stdin (e.g., from `tail -f` or `kubectl logs`)

```bash
kubectl logs deployment/agent --tail=0 --follow \
  | python -m eval.realtime_demo --stdin --json-only --summary
```

### Summaries only (no JSON/table)

```bash
python -m eval.realtime_demo traces/run.jsonl --summary-only
```

Summary lines are written to stderr so you can redirect stdout to other tools without mixing formats.

### Disable metrics when not needed

```bash
python -m eval.realtime_demo traces/run.jsonl --json-only --no-metrics
```

Prometheus/OTel sinks are disabled, which is useful for air-gapped testing environments.

For every run, AMDM can persist monitor state (`--state`/`--state-interval`) so restarts resume from the latest EWMA/covariance baselines instead of recomputing from scratch.

---

## JSON payload structure

Each turn is emitted as a JSON object:

```jsonc
{
  "agent_id": "agent-1",
  "turn_id": 4,
  "score": 4.03,
  "threshold": 2.4,
  "is_alerting": true,
  "events": [
    {
      "detector": "goal_drift",
      "turn_id": 4,
      "score": 4.03,
      "threshold": 2.4,
      "severity": "high",
      "message": "Goal drift suspected…",
      "agent_id": "agent-1"
    }
  ],
  "features": { "tokens_in": 180, ... },        // optional
  "raw": { "trace_id": "trace-1", ... }        // optional (redactable)
}
```

See [docs/JSON_PAYLOAD.md](docs/JSON_PAYLOAD.md) for the JSON schema, field descriptions, and redaction options.

---

## Development workflow

```bash
pytest -q                     # Run unit tests
python -m eval.offline_eval   # Rebuild evaluation artifacts
python -m eval.realtime_demo examples/sample_traces/agents_trace.jsonl --json-only --summary
```

- Configure log verbosity via `--log-level` (`critical|error|warning|info|debug`).
- Skip metrics with `--no-metrics` when Prometheus/OTel exporters are unavailable.
- Use `--poll-interval` and `--follow` for long-running tail operations.

If you are contributing enhancements: lint, add tests, and ensure new CLI options are documented under `docs/`.

---

## Troubleshooting tips

| Symptom | Suggestion |
|---------|------------|
| No events emitted but score is high | Enable `--include-features --include-raw` to inspect feature values and raw payloads; verify detectors thresholds in `amdm/detectors.py`. |
| JSON output mixes with tables | Add `--json-only` (or `--summary-only`) to suppress tables, and redirect stdout/stderr appropriately. |
| Schema mismatches after upgrading | Regenerate state files or run with `--state` pointing to a new path. The CLI warns when feature columns change. |
| Need to redact sensitive text | Use `--include-raw --redact-raw` so goal/plan/action fields are masked. |
| Metrics server not needed | Add `--no-metrics` to skip Prometheus/OTel exporters. |

---

## License

Distributed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

Happy monitoring! If you build new detectors, simulators, or dashboards on top of AMDM, please share back via issues or pull requests.
