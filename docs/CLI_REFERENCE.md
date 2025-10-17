# CLI Reference

## `python -m eval.offline_eval`

Runs the synthetic scenarios defined in `sim/scenarios.yaml` and reports detector performance.

### Outputs

| Artifact | Purpose |
|----------|---------|
| `eval/eval_report.md` | Human-readable summary (precision/recall/F1, ROC/PR metrics) |
| `eval/metrics.csv` | Tabular metrics for spreadsheets or BI tools |
| `eval/scenario_metrics.csv` | Optional per-scenario metrics |
| `eval/roc_*.png`, `eval/pr_*.png` | ROC/PR visualisations |

### Flags

The offline harness currently takes no flags; edit `sim/scenarios.yaml` or detector configuration inside `eval/offline_eval.py` to explore alternative setups.

---

## `python -m eval.realtime_demo`

Streams AMDM scores and detector events from a trace file or stdin. Supports incremental tailing, state persistence, metric export, and multiple output formats.

### Core options

| Flag | Description |
|------|-------------|
| `input` | Path to JSONL trace (Agents SDK or OTel); omit when using `--stdin` |
| `--stdin` | Read trace lines from stdin; auto-detects source type |
| `--follow` | Poll the input file for appended lines |
| `--poll-interval` | Seconds between polls when `--follow` is set |
| `--sleep` | Delay between processed turns (emulates wall-clock pacing) |
| `--state PATH` | Persist AMDM monitor state to JSON (loads on startup if present) |
| `--state-interval N` | Persist state every `N` events instead of every turn |
| `--no-metrics` | Disable Prometheus/OTel metric sinks |
| `--prometheus-port` | Expose Prometheus metrics on the given port |
| `--otel-metrics PATH` | Write OTEL-friendly NDJSON metrics to the given path |
| `--log-level LEVEL` | Set logging verbosity (`critical|error|warning|info|debug`) |

### Output controls

| Flag | Description |
|------|-------------|
| `--json-only` | Emit one NDJSON object per turn (suppresses table output) |
| `--json-pretty` | Pretty-print JSON when `--json-only` is active |
| `--include-features` | Embed normalized feature vector in the JSON payload |
| `--include-raw` | Embed sanitized raw record in the JSON payload |
| `--redact-raw` | Mask goal/plan/action text inside the raw payload |
| `--summary` | Emit a concise summary line (to stderr) alongside other outputs |
| `--summary-only` | Emit only the summary line (stderr) without JSON/tables |
| `--json-schema` | Print the per-turn JSON schema and exit |
| `--output PATH` | Optional JSONL file path to receive per-turn payloads |

### Example recipes

**Interactive table view while tailing a file**

```bash
python -m eval.realtime_demo traces/run.jsonl --follow --poll-interval 1.0
```

**Machine-readable stream with features and redacted raw payloads**

```bash
python -m eval.realtime_demo traces/run.jsonl \
  --json-only --json-pretty \
  --include-features --include-raw --redact-raw \
  --summary --state state.json --state-interval 20
```

**Pipe from stdin with summaries only**

```bash
kubectl logs deployment/agent --tail=0 --follow \
  | python -m eval.realtime_demo --stdin --summary-only
```

### Output schema

The JSON payload is documented in [JSON_PAYLOAD.md](JSON_PAYLOAD.md); retrieve the schema on demand:

```bash
python -m eval.realtime_demo --json-schema
```

---

## Simulator

The simulator (`sim/generator.py`) is invoked indirectly via the offline evaluation harness. To generate data programmatically:

```python
from sim.generator import load_scenarios, generate_scenario

scenarios = load_scenarios("sim/scenarios.yaml")
frame = generate_scenario(scenarios[0])
frame.to_json("synthetic.jsonl", orient="records", lines=True)
```

Modify `sim/scenarios.yaml` to introduce new anomaly regimes, adjust seeds, or vary turn counts.
