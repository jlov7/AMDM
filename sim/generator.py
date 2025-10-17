"""Synthetic multi-agent trace generator with controllable anomalies."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Scenario:
    name: str
    turns: int
    agents: int
    seed: int
    goal_drift_turns: List[int]
    tool_error_bursts: List[Tuple[int, int]]
    latency_spikes: List[int]


def load_scenarios(path: str) -> List[Scenario]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    scenarios = []
    for item in raw.get("scenarios", []):
        scenarios.append(
            Scenario(
                name=item["name"],
                turns=item.get("turns", 100),
                agents=item.get("agents", 1),
                seed=item.get("seed", 0),
                goal_drift_turns=item.get("goal_drift_turns", []),
                tool_error_bursts=[tuple(burst) for burst in item.get("tool_error_bursts", [])],
                latency_spikes=item.get("latency_spikes", []),
            )
        )
    return scenarios


def generate_scenario(scenario: Scenario) -> pd.DataFrame:
    rng = random.Random(scenario.seed)
    np_rng = np.random.default_rng(scenario.seed)
    rows: List[Dict] = []

    for agent_idx in range(scenario.agents):
        base_goal = f"Achieve objective {agent_idx}"
        plan_template = "Plan step A -> B -> C"
        for turn in range(scenario.turns):
            tokens_in = max(50, np_rng.normal(120, 15))
            tokens_out = max(40, np_rng.normal(180, 20))
            latency_ms = max(300, np_rng.normal(1200, 250))
            approvals = 1 if rng.random() > 0.1 else 0
            tool_calls = max(1, int(np_rng.poisson(2)))
            tool_errors = 0
            loop_depth = int(np_rng.integers(0, 2))

            goal_text = base_goal
            plan_text = plan_template
            action_text = "Executed step C"
            tags: List[str] = []

            if turn in scenario.goal_drift_turns:
                action_text = "Diverted to alternate objective"
                approvals = 0
                tags.append("goal_drift")

            for start, end in scenario.tool_error_bursts:
                if start <= turn <= end:
                    tool_errors = max(tool_errors, int(np_rng.integers(1, 3)))
                    tags.append("tool_error_burst")

            if turn in scenario.latency_spikes:
                latency_ms = max(latency_ms, np_rng.normal(6000, 500))
                tokens_out *= 1.8
                tags.append("cost_latency_spike")

            row = {
                "trace_id": f"scenario-{scenario.name}-agent-{agent_idx}",
                "span_id": f"{agent_idx}-{turn}",
                "agent_id": f"agent-{agent_idx}",
                "turn_id": turn,
                "timestamp": pd.Timestamp.now(),
                "tokens_in": float(tokens_in),
                "tokens_out": float(tokens_out),
                "latency_ms": float(latency_ms),
                "approval_count": float(approvals),
                "tool_calls": float(tool_calls),
                "tool_errors": float(tool_errors),
                "tool_error_rate": float(tool_errors) / float(tool_calls),
                "goal_text": goal_text,
                "plan_text": plan_text,
                "action_text": action_text,
                "loop_depth": float(loop_depth),
                "cost_usd": float(tokens_in + tokens_out) * 0.00002,
                "anomaly_tags": ",".join(sorted(set(tags))),
            }
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame.sort_values(["agent_id", "turn_id"], inplace=True, ignore_index=True)
    return frame


def generate_all(scenarios: Iterable[Scenario]) -> Dict[str, pd.DataFrame]:
    return {scenario.name: generate_scenario(scenario) for scenario in scenarios}
