"""Utilities for labeling synthetic anomalies for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class LabeledDataset:
    events: pd.DataFrame
    labels: pd.DataFrame


def build_labels(events: pd.DataFrame) -> LabeledDataset:
    """Extract anomaly annotations from simulator metadata."""

    if "anomaly_tags" not in events.columns:
        labels = pd.DataFrame({"turn_id": events.get("turn_id", []), "label": [[] for _ in range(len(events))]})
        return LabeledDataset(events=events, labels=labels)

    tags = events["anomaly_tags"].fillna("").tolist()
    label_rows: List[Dict] = []
    for turn_id, tag_str in zip(events.get("turn_id", range(len(events))), tags):
        if not tag_str:
            label_rows.append({"turn_id": turn_id, "label": []})
        else:
            label_rows.append({"turn_id": turn_id, "label": tag_str.split(",")})
    labels = pd.DataFrame(label_rows)
    return LabeledDataset(events=events, labels=labels)
