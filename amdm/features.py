"""Feature extraction from normalized turn-level events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from difflib import SequenceMatcher


def _text_delta(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - SequenceMatcher(None, a or "", b or "").ratio()


@dataclass
class FeatureSet:
    frame: pd.DataFrame
    feature_columns: List[str]


class FeatureExtractor:
    """Transform normalized events into AMDM feature vectors."""

    FEATURE_COLUMNS = [
        "tokens_in",
        "tokens_out",
        "tokens_total",
        "token_io_ratio",
        "latency_ms",
        "latency_per_token",
        "approval_count",
        "tool_error_rate",
        "tool_error_density",
        "goal_to_plan_delta",
        "plan_to_action_delta",
        "loop_depth",
        "cost_usd",
    ]

    def transform(self, events: pd.DataFrame) -> FeatureSet:
        frame = events.copy()
        frame["goal_to_plan_delta"] = [
            _text_delta(goal, plan)
            for goal, plan in zip(frame.get("goal_text", ""), frame.get("plan_text", ""))
        ]
        frame["plan_to_action_delta"] = [
            _text_delta(plan, action)
            for plan, action in zip(
                frame.get("plan_text", ""), frame.get("action_text", "")
            )
        ]
        frame["tokens_total"] = frame.get("tokens_in", 0.0) + frame.get("tokens_out", 0.0)
        frame["token_io_ratio"] = [
            (out / max(1.0, float(inp))) for inp, out in zip(frame.get("tokens_in", 0.0), frame.get("tokens_out", 0.0))
        ]
        frame["latency_per_token"] = [
            (lat / max(1.0, float(total)))
            for lat, total in zip(frame.get("latency_ms", 0.0), frame["tokens_total"])
        ]
        frame["tool_error_density"] = [
            (err / max(1.0, float(total)))
            for err, total in zip(frame.get("tool_errors", 0.0), frame["tokens_total"])
        ]
        frame.fillna(
            {
                "tokens_in": 0.0,
                "tokens_out": 0.0,
                "tokens_total": 0.0,
                "token_io_ratio": 0.0,
                "latency_ms": 0.0,
                "latency_per_token": 0.0,
                "approval_count": 0.0,
                "tool_error_rate": 0.0,
                "tool_error_density": 0.0,
                "loop_depth": 0.0,
                "cost_usd": 0.0,
            },
            inplace=True,
        )
        return FeatureSet(frame=frame, feature_columns=self.FEATURE_COLUMNS)
