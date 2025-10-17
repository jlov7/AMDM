# Adaptive Multi-Dimensional Monitoring (AMDM) Explainer

AMDM combines per-axis adaptive smoothing with multivariate distance scoring to identify anomalous behavior in agentic systems. Each turn of an agent session is represented by a feature vector capturing usage metrics (tokens, latency, approvals), control signals (loop depth, tool errors), and semantic deltas (goal→plan, plan→action divergence).

We apply an Exponentially Weighted Moving Average (EWMA) to track the baseline mean and variance of each axis. This normalizes features into z-scores that adapt to local regime shifts. A rolling window of z-score vectors forms the baseline covariance matrix. The Mahalanobis distance between the current turn and the baseline quantifies how far the agent's behavior deviates across all axes simultaneously, accounting for feature correlations.

To reduce alert flapping, we use rising/falling thresholds with hysteresis: a turn must exceed the alert band to trigger and fall below a lower recovery band to clear. Detector modules consume the AMDM score plus feature context to flag specific anomaly classes:

- **Goal drift**: sustained divergence between planned and executed actions when the AMDM score is elevated.
- **Tool error bursts**: rising tool error EWMA crossing a dynamic band.
- **Cost/latency spikes**: elevated AMDM score driven by heavy token use or latency.

For further reading, see the original AMDM paper and the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/), which inform our ingestion layer. Use `eval/offline_eval.py` to reproduce precision/recall metrics on synthetic scenarios, and `eval/realtime_demo.py` to observe streaming detections.
