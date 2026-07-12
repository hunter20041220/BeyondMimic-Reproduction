"""Composition of multiple guidance objectives."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use guidance costs") from exc


GuidanceCost = Callable[[torch.Tensor, dict[str, Any]], tuple[torch.Tensor, dict[str, torch.Tensor]]]


def composed_guidance_cost(
    predicted_trajectory: torch.Tensor,
    context: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted sum of guidance objectives."""
    objectives: list[tuple[str, GuidanceCost, dict[str, Any], float]] = context["objectives"]
    total = torch.zeros(predicted_trajectory.shape[0], dtype=predicted_trajectory.dtype, device=predicted_trajectory.device)
    diagnostics: dict[str, torch.Tensor] = {}
    for name, fn, sub_context, weight in objectives:
        cost, diag = fn(predicted_trajectory, sub_context)
        total = total + float(weight) * cost
        diagnostics[f"{name}_cost"] = cost
        for key, value in diag.items():
            diagnostics[f"{name}_{key}"] = value
    return total, diagnostics
