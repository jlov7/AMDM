"""Offline evaluation harness for AMDM detectors."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from amdm.amdm import AMDMConfig, AMDMMonitor
from amdm.detectors import (
    CostLatencySpikeDetector,
    GoalDriftDetector,
    ToolErrorBurstDetector,
)
from amdm.features import FeatureExtractor
from sim.generator import Scenario, generate_scenario, load_scenarios


OUTPUT_DIR = Path("eval")

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


@dataclass
class DetectorRecord:
    turn_id: int
    label: bool
    detected: bool
    score: float
    agent_id: str


def _ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _build_monitor(feature_names: List[str]) -> AMDMMonitor:
    weights = [FEATURE_WEIGHTS.get(name, 1.0) for name in feature_names]
    config = AMDMConfig(
        alpha=0.18,
        cov_window=80,
        min_baseline=12,
        alert_threshold=2.4,
        hysteresis=0.4,
        feature_weights=weights,
        distance_quantile=0.85,
    )
    return AMDMMonitor(feature_names=feature_names, config=config)


def _run_detectors(frame: pd.DataFrame, feature_columns: List[str]) -> Dict[str, List[DetectorRecord]]:
    monitor_by_agent: Dict[str, AMDMMonitor] = {}
    detectors_by_agent: Dict[str, Dict[str, object]] = {}
    records: Dict[str, List[DetectorRecord]] = {
        "goal_drift": [],
        "tool_error_burst": [],
        "cost_latency_spike": [],
    }

    for _, row in frame.iterrows():
        agent_label = row.get("agent_id", "agent")
        trace_id = row.get("trace_id")
        agent_key = str(trace_id or agent_label or "agent")
        monitor = monitor_by_agent.setdefault(
            agent_key, _build_monitor(feature_columns)
        )
        detector_map = detectors_by_agent.setdefault(
            agent_key,
            {
                "goal_drift": GoalDriftDetector(),
                "tool_error_burst": ToolErrorBurstDetector(),
                "cost_latency_spike": CostLatencySpikeDetector(),
            },
        )

        features = [row[col] for col in feature_columns]
        state = monitor.update(features)
        enriched = row.to_dict()
        enriched["amdm_score"] = state.last_score
        enriched["amdm_threshold"] = state.threshold

        label_tags = []
        if isinstance(row.get("anomaly_tags"), str) and row["anomaly_tags"]:
            label_tags = row["anomaly_tags"].split(",")

        for key, detector in detector_map.items():
            event = detector.update(enriched)
            label = key in label_tags
            detected = event is not None
            score = state.last_score
            records[key].append(
                DetectorRecord(
                    turn_id=int(row.get("turn_id", 0)),
                    label=label,
                    detected=bool(detected),
                    score=score,
                    agent_id=agent_key,
                )
            )
    return records


def _latency_metrics(det_records: List[DetectorRecord]) -> Dict[str, float]:
    if not det_records:
        return {
            "false_positive_rate": float("nan"),
            "latency_mean": float("nan"),
            "detection_rate": float("nan"),
        }
    false_positives = sum(1 for rec in det_records if rec.detected and not rec.label)
    negatives = sum(1 for rec in det_records if not rec.label)
    if negatives == 0:
        fpr = 0.0 if false_positives == 0 else float("inf")
    else:
        fpr = false_positives / negatives

    latencies: List[int] = []
    segments_total = 0
    records_by_agent: Dict[str, List[DetectorRecord]] = {}
    for rec in det_records:
        records_by_agent.setdefault(rec.agent_id, []).append(rec)

    for agent_records in records_by_agent.values():
        segments: List[tuple[int, int]] = []
        in_segment = False
        start = -1
        for rec in agent_records:
            if rec.label:
                if not in_segment:
                    start = rec.turn_id
                    in_segment = True
                end = rec.turn_id
            else:
                if in_segment:
                    segments.append((start, end))
                    in_segment = False
        if in_segment:
            segments.append((start, end))

        segments_total += len(segments)
        detected_turns = [rec.turn_id for rec in agent_records if rec.detected]
        for start, _ in segments:
            detection = next((turn for turn in detected_turns if turn >= start), None)
            if detection is not None:
                latencies.append(max(0, detection - start))

    detection_rate = len(latencies) / segments_total if segments_total else float("nan")
    latency_mean = float(np.mean(latencies)) if latencies else float("nan")

    return {
        "false_positive_rate": fpr,
        "latency_mean": latency_mean,
        "detection_rate": detection_rate,
    }


def _aggregate_metrics(records: Dict[str, List[DetectorRecord]]) -> pd.DataFrame:
    rows = []
    for name, det_records in records.items():
        if not det_records:
            continue
        labels = np.array([rec.label for rec in det_records], dtype=int)
        preds = np.array([rec.detected for rec in det_records], dtype=int)
        scores = np.array([rec.score for rec in det_records], dtype=float)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        try:
            roc_auc = roc_auc_score(labels, scores) if labels.sum() > 0 else np.nan
        except ValueError:
            roc_auc = np.nan
        ap = (
            average_precision_score(labels, scores)
            if labels.sum() > 0
            else np.nan
        )
        latency_stats = _latency_metrics(det_records)

        rows.append(
            {
                "detector": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "average_precision": ap,
                "false_positive_rate": latency_stats["false_positive_rate"],
                "detection_latency_mean": latency_stats["latency_mean"],
                "detection_rate": latency_stats["detection_rate"],
            }
        )
    return pd.DataFrame(rows)


def _plot_curves(name: str, records: List[DetectorRecord]):
    labels = np.array([rec.label for rec in records], dtype=int)
    scores = np.array([rec.score for rec in records], dtype=float)

    if labels.sum() == 0:
        return

    plt.figure()
    RocCurveDisplay.from_predictions(labels, scores)
    plt.title(f"ROC Curve - {name}")
    roc_path = OUTPUT_DIR / f"roc_{name}.png"
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    PrecisionRecallDisplay.from_predictions(labels, scores)
    plt.title(f"PR Curve - {name}")
    pr_path = OUTPUT_DIR / f"pr_{name}.png"
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate() -> pd.DataFrame:
    _ensure_output_dir()
    scenario_path = Path("sim/scenarios.yaml")
    scenarios = load_scenarios(str(scenario_path))
    extractor = FeatureExtractor()

    combined_records: Dict[str, List[DetectorRecord]] = {
        "goal_drift": [],
        "tool_error_burst": [],
        "cost_latency_spike": [],
    }
    per_scenario_metrics: List[pd.DataFrame] = []

    for scenario in scenarios:
        frame = generate_scenario(scenario)
        feature_set = extractor.transform(frame)
        frame = feature_set.frame
        records = _run_detectors(frame, feature_set.feature_columns)
        for key, recs in records.items():
            combined_records[key].extend(recs)
        scenario_metrics = _aggregate_metrics(records)
        if not scenario_metrics.empty:
            scenario_metrics.insert(0, "scenario", scenario.name)
            per_scenario_metrics.append(scenario_metrics)

    metrics = _aggregate_metrics(combined_records)
    metrics_path = OUTPUT_DIR / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    for key, recs in combined_records.items():
        _plot_curves(key, recs)

    report_path = OUTPUT_DIR / "eval_report.md"
    report_lines = ["# AMDM Offline Evaluation", "", "## Overall Metrics", "", metrics.to_markdown(index=False)]
    if per_scenario_metrics:
        scenario_df = pd.concat(per_scenario_metrics, ignore_index=True)
        scenario_df.to_csv(OUTPUT_DIR / "scenario_metrics.csv", index=False)
        report_lines.extend(["", "## Scenario Metrics", "", scenario_df.to_markdown(index=False)])
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    metrics = evaluate()
    print(metrics.to_string(index=False))
