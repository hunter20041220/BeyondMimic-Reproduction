#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _as_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _phase_mask(times: np.ndarray, start: float, end: float) -> np.ndarray:
    return np.logical_and(times >= start, times < end)


def _mean(arr: np.ndarray) -> float | None:
    return float(np.asarray(arr).mean()) if np.asarray(arr).size else None


def _rate(arr: np.ndarray) -> float | None:
    return float(np.asarray(arr, dtype=np.bool_).mean()) if np.asarray(arr).size else None


def _max_consecutive_seconds(mask: np.ndarray, frequency_hz: float) -> float:
    flat = np.asarray(mask, dtype=np.bool_).reshape(-1)
    best = 0
    current = 0
    for value in flat:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return float(best) / float(frequency_hz)


def _phase_summary(
    data: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    run_target_vx: float,
    frequency_hz: float,
) -> dict[str, Any]:
    actual = np.asarray(data["actual_velocity_xy_current_yaw"], dtype=np.float32)
    target = np.asarray(data["target_velocity_xy"], dtype=np.float32)
    guided = np.asarray(data.get("guided_future_velocity_xy_mean", np.zeros_like(actual)), dtype=np.float32)
    planned = np.asarray(data.get("planned_future_velocity_xy_mean", np.zeros_like(actual)), dtype=np.float32)
    phase_actual = actual[:, mask]
    phase_target = target[:, mask]
    phase_guided = guided[:, mask]
    phase_planned = planned[:, mask]
    vx = phase_actual[..., 0]
    vy = phase_actual[..., 1]
    speed = np.linalg.norm(phase_actual, axis=-1)
    directional_ok = np.logical_and(vx >= 0.8 * float(run_target_vx), np.abs(vy) < np.maximum(vx, 1.0e-6))
    out: dict[str, Any] = {
        "target_vx_mean": _mean(phase_target[..., 0]),
        "target_vy_mean": _mean(phase_target[..., 1]),
        "actual_vx_mean": _mean(vx),
        "actual_vx_max": float(vx.max()) if vx.size else None,
        "actual_vy_abs_mean": _mean(np.abs(vy)),
        "actual_speed_mean": _mean(speed),
        "guided_future_vx_mean": _mean(phase_guided[..., 0]),
        "guided_future_vy_mean": _mean(phase_guided[..., 1]),
        "planned_future_vx_mean": _mean(phase_planned[..., 0]),
        "planned_future_vy_mean": _mean(phase_planned[..., 1]),
        "directional_80pct_run_rate": _rate(directional_ok),
        "directional_80pct_run_max_consecutive_s": _max_consecutive_seconds(directional_ok[0], frequency_hz)
        if directional_ok.ndim == 2 and directional_ok.shape[0]
        else None,
    }
    for key in ["flight_phase_proxy", "physical_fall", "physical_illegal_contact", "physical_accepted"]:
        if key in data:
            out[f"{key}_rate"] = _rate(np.asarray(data[key])[:, mask])
    if "root_height" in data:
        root = np.asarray(data["root_height"], dtype=np.float32)[:, mask]
        out["root_height_min"] = float(root.min()) if root.size else None
        out["root_height_mean"] = _mean(root)
    if "foot_contact" in data:
        contact = np.asarray(data["foot_contact"], dtype=np.bool_)[:, mask]
        out["foot_contact_rate_per_foot"] = contact.mean(axis=(0, 1)).astype(float).tolist() if contact.size else []
    return out


def _polyline(points: np.ndarray, *, width: int, height: int, x0: int, y0: int, y_min: float, y_max: float) -> str:
    if points.size == 0:
        return ""
    denom = max(1.0e-6, float(y_max - y_min))
    xs = np.linspace(float(x0), float(x0 + width), points.shape[0])
    ys = float(y0 + height) - (points.astype(np.float32) - float(y_min)) / denom * float(height)
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def _write_svg_plot(data: dict[str, np.ndarray], output_svg: Path, frequency_hz: float) -> Path:
    actual = np.asarray(data["actual_velocity_xy_current_yaw"], dtype=np.float32)[0]
    target = np.asarray(data["target_velocity_xy"], dtype=np.float32)[0]
    guided = np.asarray(data.get("guided_future_velocity_xy_mean", np.zeros_like(actual[None])), dtype=np.float32)[0]
    planned = np.asarray(data.get("planned_future_velocity_xy_mean", np.zeros_like(actual[None])), dtype=np.float32)[0]
    flight = np.asarray(data.get("flight_phase_proxy", np.zeros(actual.shape[:1])[None]), dtype=np.float32)[0]
    root = np.asarray(data.get("root_height", np.zeros(actual.shape[:1])[None]), dtype=np.float32)[0]
    width = 1100
    panel_h = 150
    left = 70
    inner_w = 980
    panels = [
        ("vx m/s", [("cmd", target[:, 0], "#111111"), ("actual", actual[:, 0], "#1f77b4"), ("guided", guided[:, 0], "#2ca02c"), ("planned", planned[:, 0], "#ff7f0e")]),
        ("vy m/s", [("cmd", target[:, 1], "#111111"), ("actual", actual[:, 1], "#d62728"), ("guided", guided[:, 1], "#9467bd")]),
        ("flight", [("flight", flight.reshape(-1), "#17becf")]),
        ("root h", [("root", root.reshape(-1), "#8c564b")]),
    ]
    total_h = 40 + len(panels) * (panel_h + 35)
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_h}" viewBox="0 0 {width} {total_h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial, sans-serif;font-size:13px}.axis{stroke:#999;stroke-width:1}.grid{stroke:#e5e5e5;stroke-width:1}.line{fill:none;stroke-width:2}</style>',
        f'<text x="{left}" y="24">velocity guidance rollout, {actual.shape[0] / float(frequency_hz):.2f}s at {frequency_hz:.1f}Hz</text>',
    ]
    y = 45
    for title, series in panels:
        values = np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for _, v, _ in series])
        y_min = float(np.nanmin(values)) if values.size else 0.0
        y_max = float(np.nanmax(values)) if values.size else 1.0
        if abs(y_max - y_min) < 1.0e-6:
            y_min -= 1.0
            y_max += 1.0
        pad = 0.05 * (y_max - y_min)
        y_min -= pad
        y_max += pad
        svg.append(f'<text x="10" y="{y + 18}">{title}</text>')
        svg.append(f'<line class="axis" x1="{left}" y1="{y + panel_h}" x2="{left + inner_w}" y2="{y + panel_h}"/>')
        svg.append(f'<line class="axis" x1="{left}" y1="{y}" x2="{left}" y2="{y + panel_h}"/>')
        for frac in (0.25, 0.5, 0.75):
            gy = y + panel_h * frac
            svg.append(f'<line class="grid" x1="{left}" y1="{gy:.1f}" x2="{left + inner_w}" y2="{gy:.1f}"/>')
        legend_x = left + 8
        for label, values_i, color in series:
            points = _polyline(np.asarray(values_i).reshape(-1), width=inner_w, height=panel_h, x0=left, y0=y, y_min=y_min, y_max=y_max)
            svg.append(f'<polyline class="line" stroke="{color}" points="{points}"/>')
            svg.append(f'<text x="{legend_x}" y="{y + 15}" fill="{color}">{label}</text>')
            legend_x += 80
        svg.append(f'<text x="{left + inner_w + 8}" y="{y + 12}">{y_max:.2f}</text>')
        svg.append(f'<text x="{left + inner_w + 8}" y="{y + panel_h}">{y_min:.2f}</text>')
        y += panel_h + 35
    svg.append("</svg>")
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return output_svg


def _maybe_plot(data: dict[str, np.ndarray], output_plot: Path, frequency_hz: float) -> tuple[Path, bool, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        svg_path = output_plot if output_plot.suffix.lower() == ".svg" else output_plot.with_suffix(".svg")
        return _write_svg_plot(data, svg_path, frequency_hz), True, "svg"

    actual = np.asarray(data["actual_velocity_xy_current_yaw"], dtype=np.float32)[0]
    target = np.asarray(data["target_velocity_xy"], dtype=np.float32)[0]
    guided = np.asarray(data.get("guided_future_velocity_xy_mean", np.zeros_like(actual[None])), dtype=np.float32)[0]
    planned = np.asarray(data.get("planned_future_velocity_xy_mean", np.zeros_like(actual[None])), dtype=np.float32)[0]
    t = np.arange(actual.shape[0], dtype=np.float32) / float(frequency_hz)
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t, target[:, 0], label="cmd vx", color="black")
    axes[0].plot(t, actual[:, 0], label="actual vx", color="#1f77b4")
    axes[0].plot(t, guided[:, 0], label="guided future vx", color="#2ca02c", alpha=0.8)
    axes[0].plot(t, planned[:, 0], label="planned future vx", color="#ff7f0e", alpha=0.7)
    axes[0].set_ylabel("vx m/s")
    axes[0].legend(loc="best", ncol=2, fontsize=8)
    axes[1].plot(t, target[:, 1], label="cmd vy", color="black")
    axes[1].plot(t, actual[:, 1], label="actual vy", color="#d62728")
    axes[1].plot(t, guided[:, 1], label="guided future vy", color="#9467bd", alpha=0.8)
    axes[1].set_ylabel("vy m/s")
    axes[1].legend(loc="best", ncol=2, fontsize=8)
    if "flight_phase_proxy" in data:
        axes[2].plot(t, np.asarray(data["flight_phase_proxy"], dtype=np.float32)[0], label="flight proxy")
    if "foot_contact" in data:
        contact = np.asarray(data["foot_contact"], dtype=np.float32)[0]
        axes[2].plot(t, contact.mean(axis=-1), label="mean foot contact", alpha=0.8)
    axes[2].set_ylabel("contact")
    axes[2].legend(loc="best", fontsize=8)
    if "root_height" in data:
        axes[3].plot(t, np.asarray(data["root_height"], dtype=np.float32)[0], label="root height")
    if "physical_fall" in data:
        axes[3].plot(t, np.asarray(data["physical_fall"], dtype=np.float32)[0], label="physical fall", alpha=0.7)
    axes[3].set_ylabel("height/fall")
    axes[3].set_xlabel("time s")
    axes[3].legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_plot, dpi=160)
    plt.close(fig)
    return output_plot, True, "matplotlib"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize velocity-guided Isaac rollout NPZ.")
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-png")
    parser.add_argument("--frequency-hz", type=float, default=50.0)
    parser.add_argument("--run-target-vx", type=float, required=True)
    parser.add_argument("--schedule", choices=["walk_to_run", "walk_run_walk"], default="walk_to_run")
    args = parser.parse_args()

    rollout = Path(args.rollout)
    with np.load(rollout, allow_pickle=False) as npz:
        data = {key: npz[key] for key in npz.files if key != "metadata_json"}
    steps = int(np.asarray(data["actual_velocity_xy_current_yaw"]).shape[1])
    times = np.arange(steps, dtype=np.float32) / float(args.frequency_hz)
    if args.schedule == "walk_to_run":
        phases = {
            "walk_0_4s": _phase_mask(times, 0.0, 4.0),
            "ramp_4_11s": _phase_mask(times, 4.0, 11.0),
            "run_plateau_11_20s": _phase_mask(times, 11.0, times[-1] + 1.0 / float(args.frequency_hz)),
        }
    else:
        phases = {
            "walk_start_0_4s": _phase_mask(times, 0.0, 4.0),
            "accel_4_10s": _phase_mask(times, 4.0, 10.0),
            "run_10_16s": _phase_mask(times, 10.0, 16.0),
            "decel_16_22s": _phase_mask(times, 16.0, 22.0),
            "walk_end_22_25s": _phase_mask(times, 22.0, times[-1] + 1.0 / float(args.frequency_hz)),
        }
    summary = {
        "rollout": str(rollout),
        "frequency_hz": float(args.frequency_hz),
        "steps": steps,
        "duration_s": steps / float(args.frequency_hz),
        "schedule": args.schedule,
        "run_target_vx": float(args.run_target_vx),
        "success_threshold_vx_80pct": 0.8 * float(args.run_target_vx),
        "phases": {
            name: _phase_summary(data, mask, run_target_vx=args.run_target_vx, frequency_hz=args.frequency_hz)
            for name, mask in phases.items()
        },
        "overall": _phase_summary(
            data,
            np.ones_like(times, dtype=np.bool_),
            run_target_vx=args.run_target_vx,
            frequency_hz=args.frequency_hz,
        ),
    }
    if args.output_png:
        plot_path, written, backend = _maybe_plot(data, Path(args.output_png), args.frequency_hz)
        summary["plot_path"] = str(plot_path)
        summary["plot_written"] = written
        summary["plot_backend"] = backend
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
