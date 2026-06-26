#!/usr/bin/env python3
"""Evaluate joystick, waypoint, obstacle, and inpainting guidance costs."""

from __future__ import annotations

import argparse

import numpy as np

from beyondmimic_repro.data.state_latent import load_state_latent_tokens
from beyondmimic_repro.guidance.sampler import gradient_guidance_step
from beyondmimic_repro.guidance.tasks import evaluate_guidance_suite, joystick_cost, obstacle_cost, waypoint_cost
from beyondmimic_repro.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/state_latent/train_windows.npz")
    parser.add_argument("--metrics", default="outputs/metrics/guidance_metrics.json")
    parser.add_argument("--step-size", type=float, default=0.05)
    args = parser.parse_args()
    tokens = load_state_latent_tokens(args.dataset)[:8]
    base = evaluate_guidance_suite(tokens)
    guided, before, after = gradient_guidance_step(
        tokens,
        lambda x: waypoint_cost(x, np.array([1.0, 0.0], dtype=np.float64))
        + 0.1 * obstacle_cost(x, np.array([0.5, 0.0], dtype=np.float64), radius=0.2)
        + 0.1 * joystick_cost(x, np.array([0.03, 0.0], dtype=np.float64)),
        step_size=args.step_size,
        guided_dims=2,
    )
    guided_metrics = evaluate_guidance_suite(guided)
    summary = {
        "status": "ok",
        "dataset": args.dataset,
        "unguided": base,
        "guided": guided_metrics,
        "composed_cost_before": before,
        "composed_cost_after": after,
    }
    write_json(args.metrics, summary)
    print(summary)


if __name__ == "__main__":
    main()
