"""Seeded toy evaluation with an optional AIGov Core evidence event."""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean


ENVIRONMENT_VERSION = "nonlinear-target-v1"
MODEL_VERSION = "linear-surrogate-v1"
DATASET_ID = "synthetic-control-trajectories-v1"
AI_SYSTEM_ID = "model-based-eval-suite"


def real_step(state: float, action: float) -> float:
    return 0.9 * state + action - 0.04 * state * state


def learned_step(state: float, action: float) -> float:
    return 0.9 * state + action


def aggressive(state: float, step: int) -> float:
    del state, step
    return 1.0


def conservative(state: float, step: int) -> float:
    del step
    return max(-0.2, min(0.2, 1.0 - state))


POLICIES = {"aggressive": aggressive, "conservative": conservative}


def evaluate(policy_name: str, seed: int, horizon: int = 30) -> dict:
    if policy_name not in POLICIES:
        raise ValueError(f"unknown policy: {policy_name}")
    if horizon < 1:
        raise ValueError("horizon must be positive")

    rng = random.Random(seed)
    initial_state = rng.uniform(-0.5, 0.5)
    model_state = actual_state = initial_state
    predicted_return = actual_return = 0.0
    transition_errors: list[float] = []

    for step in range(horizon):
        action = POLICIES[policy_name](model_state, step)
        model_next = learned_step(model_state, action)
        actual_next = real_step(actual_state, action)
        transition_errors.append(abs(model_next - actual_next))
        predicted_return -= abs(1.0 - model_next)
        actual_return -= abs(1.0 - actual_next)
        model_state, actual_state = model_next, actual_next

    return {
        "policy": policy_name,
        "seed": seed,
        "horizon": horizon,
        "initial_state": initial_state,
        "predicted_return": predicted_return,
        "actual_return": actual_return,
        "return_gap": predicted_return - actual_return,
        "mean_transition_error": fmean(transition_errors),
    }


def evaluate_suite(seeds: list[int], horizon: int = 30) -> dict:
    if not seeds:
        raise ValueError("provide at least one evaluation seed")
    episodes = [
        evaluate(policy_name, seed, horizon)
        for seed in seeds
        for policy_name in POLICIES
    ]
    summaries = {}
    for policy_name in POLICIES:
        rows = [row for row in episodes if row["policy"] == policy_name]
        summaries[policy_name] = {
            key: fmean(row[key] for row in rows)
            for key in (
                "predicted_return",
                "actual_return",
                "return_gap",
                "mean_transition_error",
            )
        }
    return {
        "environment_version": ENVIRONMENT_VERSION,
        "model_version": MODEL_VERSION,
        "evaluation_seeds": seeds,
        "horizon": horizon,
        "episodes": episodes,
        "policy_summaries": summaries,
        "limitations": [
            "Synthetic one-dimensional dynamics and a deliberately misspecified linear model.",
            "Only two fixed policies and one return definition are evaluated.",
            "This experiment is not evidence of real-world planning quality or model safety.",
        ],
    }


def build_aigov_event(
    report: dict,
    run_id: str,
    threshold: float = 0.25,
    timestamp: str | None = None,
) -> dict:
    """Create an evaluation_reported event accepted by AIGov Core's public schema."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    errors = [row["mean_transition_error"] for row in report["episodes"]]
    value = fmean(errors)
    return {
        "event_id": f"{run_id}-evaluation",
        "event_type": "evaluation_reported",
        "ts_utc": timestamp or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "actor": "model-based-eval-suite",
        "system": "world-model-reference-experiment",
        "run_id": run_id,
        "payload": {
            "ai_system_id": AI_SYSTEM_ID,
            "dataset_id": DATASET_ID,
            "model_version_id": MODEL_VERSION,
            "metric": "mean_transition_error",
            "value": value,
            "threshold": threshold,
            "passed": value <= threshold,
            "environment_version": report["environment_version"],
            "evaluation_seeds": report["evaluation_seeds"],
            "policy_summaries": report["policy_summaries"],
            "limitations": report["limitations"],
        },
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--output", type=Path, help="optional JSON evaluation report path")
    parser.add_argument("--aigov-event", type=Path, help="optional AIGov Core event path")
    parser.add_argument("--run-id", default="world-model-demo")
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    report = evaluate_suite(args.seeds, args.horizon)
    if args.output:
        write_json(args.output, report)
    if args.aigov_event:
        event = build_aigov_event(report, args.run_id, args.threshold)
        write_json(args.aigov_event, event)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
